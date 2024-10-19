import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
import argparse
import os
from glob import glob
from time import time
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import random

import sys
sys.path.append(os.path.dirname(sys.path[0]))
from hparams import get_vqgan_hparams
from bae.binaryae import BinaryGAN, BinaryAutoEncoder, load_pretrain
from llama.load_bigr import load_bigr

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

def enrich(model, bae, save, args, inp, mask, seed=0, image_size=256, num_classes=1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # # Load model:
    latent_size = image_size // 16
    
    model.eval()  # important!
    bae.eval()
    
    # Labels to condition the model with (feel free to change):
    class_labels = [1000]
    y = torch.tensor(class_labels, device=device)
    bs = y.shape[0]
    
    repeat = 4
    # Setup classifier-free guidance:
    for si in range(repeat):
        torch.manual_seed(seed+si)
        np.random.seed(seed+si)
        random.seed(seed+si)
        start_time = time()
        inp = model.supres_box(cond=y, max_new_tokens=latent_size ** 2, input_seq=inp, high_res=latent_size, ds_rate=args.downsample_rate, cond_padding=args.cls_token_num, num_iter=args.num_sample_iter,
                        out_dim=bae.codebook_size, cfg_scale=args.cfg_scale, cfg_schedule=args.cfg_schedule,
                        gumbel_temp=args.gumbel_temp, gumbel_schedule=args.gumbel_schedule, sample_logits=True, proj_emb=None)
        
        samples = inp
        end_time = time()
        print("Sample time: {}".format(end_time-start_time))
        
        samples = samples.float().transpose(1,2).reshape(bs, -1, latent_size, latent_size)
        samples = bae.decode(samples)
        
        save_i = save.rsplit('.', 1)[0] + f'_{si}.png'
        save_image(samples, save_i, nrow=4, normalize=True, value_range=(0, 1))
    
    del model, bae
    
def main(args, args_ae):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert args.ckpt
    
    #### BiGR model ####
    model = load_bigr(args, args_ae, device)
    
    ############ Binary VAE ##############
    binaryae = BinaryAutoEncoder(args_ae).to(device)
    binaryae = load_pretrain(binaryae, args.ckpt_bae)
    
    print(f"The code length of B-AE is set to {args_ae.codebook_size}")
    print(f"We load B-AE checkpoint from {args.ckpt_bae}")
    ######################################
        
    os.makedirs(args.save_path, exist_ok=True)
    sample_index = len(glob(f"{args.save_path}/*"))
    save_folder = os.path.join(args.save_path, f"{sample_index:03d}")
        
    os.makedirs(save_folder, exist_ok=True)
    save = os.path.join(save_folder, 
                    f"{sample_index:03d}_cfg{args.cfg_scale}_temp{args.temperature}_gumbel{args.gumbel_temp}_iter{args.num_sample_iter}.png")
    print("Saving to {}".format(save))
    
    ### prepare image data for inpainting
    image = Image.open(args.input_image)
    
    ds_size = args.image_size // args.downsample_rate
    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, ds_size)),
        transforms.ToTensor(),
    ])
    input_img = transform(image).to(device)
    
    input_img = F.interpolate(input_img.unsqueeze(0), size=(args.image_size, args.image_size), mode='bilinear')
    mask = torch.ones((1, 256)).to(torch.bool)
    
    with torch.no_grad():
        # Map input images to latent space + normalize latents:
        quant, binary, binary_det = binaryae.encode(input_img)
        bs, dim_b, ph, pw = binary.shape
        
        binary_cache = binary
        inp = binary.reshape(bs, dim_b, -1).transpose(1,2)
        
        ### save decoded image
        save_recon = os.path.join(save_folder, f"{sample_index:03d}_recon.png")
        samples_recon = binaryae.decode(binary_cache)
        save_image(samples_recon, save_recon, nrow=4, normalize=True, value_range=(0, 1))

        ### save original image
        save_org = os.path.join(save_folder, f"{sample_index:03d}_org.png")
        save_image(input_img, save_org, nrow=4, normalize=True, value_range=(0, 1))
        
    enrich(
        model, binaryae, save, args,
        inp=inp, mask=mask,
        seed=args.seed, image_size=args.image_size, num_classes=args.num_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GPT-L")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--save-path", type=str, default="samples_inpaint")
    parser.add_argument("--ckpt_bae", type=str, required=True, help='checkpoint path for bae tokenizer')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, required=True)
    
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
    
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--downsample_rate", type=int, default=2)
    
    args_ae = get_vqgan_hparams(parser)
    args = parser.parse_args()
    
    args_ae.img_size = args.image_size
    
    main(args, args_ae)
