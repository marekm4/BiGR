import torch
from llama.gpt import BIGR_models

def load_bigr(args, args_ae, device):
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p

    latent_size = args.image_size // 16

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

    ckpt_path = args.ckpt
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=True)
    model.load_state_dict(checkpoint["ema"])
    
    return model