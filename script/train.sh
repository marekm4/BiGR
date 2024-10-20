TORCH_DISTRIBUTED_DEBUG=INFO TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 torchrun --nnodes=1 --nproc_per_node=8 train_ddp.py \
 --ckpt_bae ckpts/bae/bae_d24/binaryae_ema_1000000.th \
 --dataset custom --codebook_size 24 --img_size 256 --norm_first \
 --model BiGR-L \
 --data-path \
 --num-classes 1000 \
 --global-batch-size 512 \
 --ckpt-every 50000 \
 --epochs 10000 \
 --mixed-precision bf16 \
 --weight_decay 2e-2 \
 --p_flip \
 --focal 0.0 \
 --cfg-scale 4.0 \
 --temperature 1.0 \
 --use_adaLN