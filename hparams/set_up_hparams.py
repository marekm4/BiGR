import argparse
from .defaults.binarygan_default import HparamsBinaryAE, add_vqgan_args

def apply_parser_values_to_H(H, args):
    # NOTE default args in H will be overwritten by any default parser args
    args = args.__dict__
    for arg in args:
        if args[arg] is not None:
            H[arg] = args[arg]

    return H

def get_vqgan_parser_args(parser):
    add_vqgan_args(parser)
    parser_args = parser.parse_args()
    
    return parser_args

def args2H(args):
    H = HparamsBinaryAE(args.dataset)
    H = apply_parser_values_to_H(H, args)

    if not H.lr:
        H.lr = H.base_lr * H.batch_size

    return H

def get_vqgan_hparams(parser):
    add_vqgan_args(parser)
    parser_args = parser.parse_args()
    H = HparamsBinaryAE(parser_args.dataset)
    H = apply_parser_values_to_H(H, parser_args)

    if not H.lr:
        H.lr = H.base_lr * H.batch_size

    return H
