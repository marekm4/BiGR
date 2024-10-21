# Modified from:
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/train.py
#   LlamaGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/train/train_c2i.py

import torch
# the first flag bteelow was False when we sted this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from llama.gpt import BIGR_models
from sample import sample_func
from torchvision.utils import save_image
from bae.binaryae import BinaryAutoEncoder, load_pretrain
from hparams import get_vqgan_hparams
from data import ImageNetPrepareBAEDataset
import torch._dynamo
torch._dynamo.config.suppress_errors = True

#################################################################################
#                               Set Weight Decay                                #
#################################################################################
def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'denoise_mlp' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
    
#################################################################################
#                               Adjust Learning Rate                            #
#################################################################################

def get_lr(it, learning_rate, min_lr, warmup_iters, lr_decay_iters):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
   
#################################################################################
#                                  Training Loop                                #
#################################################################################
def init_dist(launcher="pytorch", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank

def main(args, args_ae):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP
    # dist.init_process_group("nccl")
    # Initialize distributed training
    local_rank      = init_dist()
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0
    # dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=5000))
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    device = local_rank
    seed = args.global_seed + global_rank
    
    # seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={global_rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        save_dir = f"{experiment_dir}/sample_images"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        for k in args.__dict__:
            logger.info(k + ": " + str(args.__dict__[k]))
    else:
        logger = create_logger(None)

    ##### Create model:
    assert args.image_size % 16 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 16
    
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
     
    model = BIGR_models[args.model](
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        binary_size=args_ae.codebook_size,
        p_flip=args.p_flip,
        n_repeat=args.n_repeat,
        aux=args.aux,
        focal=args.focal,
        sample_temperature=args.temperature,
        n_sample_steps=args.n_sample_steps,
        infer_steps = args.infer_steps,
        use_adaLN = args.use_adaLN,
        seq_len = args.seq_len,
        alpha = args.alpha
    ).to(device)
    
    if args.resume is not None:
        resume_path = args.resume
        print("Resuming from checkpoint: {}".format(args.resume))
        resume_ckpt = torch.load(resume_path, map_location=lambda storage, loc: storage, weights_only=True)
        model.load_state_dict(resume_ckpt["model"])
    ####################
    
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    if args.resume is not None:
        ema.load_state_dict(resume_ckpt["ema"])
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    
    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0
        
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    
    ############ Binary VAE ##############
    binaryae = BinaryAutoEncoder(args_ae).to(device)
    binaryae = load_pretrain(binaryae, args.ckpt_bae)
    
    print(f"The code length of B-AE is set to {args_ae.codebook_size}")
    print(f"We load B-AE checkpoint from {args.ckpt_bae}")
    ######################################
    
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"MLP Parameters in GPT: {sum(p.numel() for p in model.module.denoise_mlp.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    param_groups = add_weight_decay(model.module, weight_decay=args.weight_decay)
    opt = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(opt)
    
    if args.resume is not None:
        opt.load_state_dict(resume_ckpt["opt"])
    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # dataset = mxImageNetDataset(
    #     rec_file=args.data_path,
    #     to_rgb=True,
    #     transform=transform
    # )
    # print("We are using mxnet record!")
    
    if not args.use_prepared_data:
        dataset = ImageFolder(args.data_path, transform=transform)
    else:
        dataset = ImageNetPrepareBAEDataset(
            bin_dir=args.data_path,
            codebook=args_ae.codebook_size,
            latent_res=int(args.seq_len**0.5)
            )
        
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=global_rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
        
    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    binaryae.eval()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_bce_loss = 0
    running_acc = 0
    start_time = time()

    if args.resume is not None:
        resume_iter = int(args.resume.split('/')[-1].split('.')[0])
        train_steps = resume_iter
        print("We resume training from {} steps.".format(resume_iter))

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            # constant learning rate
            lr = get_lr(train_steps, learning_rate=args.lr, min_lr=args.lr, warmup_iters=args.warmup, lr_decay_iters=-1)
            opt.param_groups[0]['lr'] = lr
                    
            x = x.to(device)
            y = y.to(device)
            
            if args.use_prepared_data:
                bs, dim, ph, pw = x.shape
                inp = x.reshape(bs, dim, -1).transpose(1,2)
                trg = x.reshape(bs, dim, -1).transpose(1,2)
            else:
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    quant, binary, binary_det = binaryae.encode(x)
                
                bs, dim_b, ph, pw = binary.shape
                    
                # process inputs and targets
                inp = binary.reshape(bs, dim_b, -1).transpose(1,2)
                trg = binary.reshape(bs, dim_b, -1).transpose(1,2)
            
            with torch.cuda.amp.autocast(dtype=ptdtype):
                _, stats = model(inp=inp, cond_idx=y, targets=trg)
            
            loss = stats['loss']
            # model.module.clear_cond()
            
            opt.zero_grad()
            # loss.backward()
            scaler.scale(loss).backward()
            
            if args.max_grad_norm != 0.0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
            # opt.step()
            scaler.step(opt)
            scaler.update()
            # update_ema(ema, model.module)
            update_ema(ema, model.module._orig_mod if not args.no_compile else model.module)

            # Log loss values:
            running_loss += loss.item()
            running_bce_loss += stats['bce_loss'].item()
            running_acc += stats['acc'].item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0  or train_steps == 1:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_bce_loss = torch.tensor(running_bce_loss / log_steps, device=device)
                avg_acc = torch.tensor(running_acc / log_steps, device=device)
                
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_bce_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_acc, op=dist.ReduceOp.SUM)
                
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_bce_loss = avg_bce_loss.item() / dist.get_world_size()
                avg_acc = avg_acc.item() / dist.get_world_size()
                
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train BCE Loss: {avg_bce_loss:.4f}, Train ACC: {avg_acc:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, lr: {opt.param_groups[0]['lr']:.6f}")
                # Reset monitoring variables:
                running_loss = 0
                running_bce_loss = 0
                running_acc = 0
                log_steps = 0
                start_time = time()

            # Save BiGR checkpoint:
            if train_steps % args.ckpt_every == 0 or train_steps == 1:
                if global_rank == 0:
                    ###### save checkpoints #####
                    if not args.no_compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()  
                        
                    checkpoint = {
                        "model": model_weight,
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    #############################
                    
                    ###### generate images ######
                    save_path = f"{save_dir}/{train_steps:07d}.png"
                    
                    model_module = model.module._orig_mod if not args.no_compile else model.module
                    model_without_ddp = deepcopy(model_module)
                    with torch.no_grad():
                        sample_func(
                            model_without_ddp, binaryae, save_path, args,
                            num_classes=args.num_classes, image_size=args.image_size)
                    ###############################
                    
                dist.barrier()
                
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # global config 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--weight_decay", type=float, default=2e-2)
    parser.add_argument("--ckpt_bae", type=str, required=True, help='checkpoint path for bae tokenizer')
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_prepared_data", action='store_true')
    parser.add_argument("--warmup", type=int, default=0)
    
    ### GPT hparams
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.0, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--use_adaLN", action='store_true')
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--p_flip", action='store_true', help='predict z0 or z0 XOR zt (flipping)')
    parser.add_argument("--focal", type=float, default=-1, help='focal coefficient')
    parser.add_argument("--alpha", type=float, default=-1, help='alpha coefficient')
    parser.add_argument("--aux", type=float, default=0.0, help='vlb weight')
    parser.add_argument("--n_repeat", type=int, default=1, help='sample timesteps n_repeat times')
    parser.add_argument("--n_sample_steps", type=int, default=256, help="time steps to sample in diffusion training")
    parser.add_argument("--seq_len", type=int, default=256)
    
    ### sample config
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--num_sample_iter", type=int, default=10)
    parser.add_argument("--gumbel_temp", type=float, default=0.)
    parser.add_argument("--cfg_schedule", type=str, default='constant', choices=['constant', 'linear'])
    parser.add_argument("--infer_steps", type=int, default=100, help="time steps to sample in diffusion inference")
    parser.add_argument("--gumbel_schedule", type=str, default='constant', choices=['constant', 'down', 'up'])
    
    args_ae = get_vqgan_hparams(parser)
    args = parser.parse_args()
    
    args_ae.img_size = args.image_size
    
    main(args, args_ae)
