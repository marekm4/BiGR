# model used
CODE=32
CKPT_BAE=ckpts/bae/bae_d32_512/binaryae_ema_720000.th
CKPT_BIGR=ckpts/gpt/bigr_L_d32_512.pt
# sample hparams
CFG=3.0
ITER=20
GUMBEL=0.01
# sample script
python sample.py \
 --model BiGR-L-512 \
 --ckpt_bae $CKPT_BAE \
 --dataset custom --codebook_size $CODE --img_size 512 --norm_first \
 --image-size 512 --seq_len 1024 \
 --ckpt $CKPT_BIGR \
 --num-classes 1000 \
 --seed 1 \
 --p_flip \
 --focal 0.0 \
 --use_adaLN \
 --cfg-scale $CFG \
 --num_sample_iter $ITER \
 --gumbel_temp $GUMBEL \
 --cfg_schedule constant \
 --save-path samples