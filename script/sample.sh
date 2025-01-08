# model used
CODE=24
CKPT_BAE=ckpts/bae/bae_d24/binaryae_ema_1000000.th
CKPT_BIGR=ckpts/gpt/bigr_L_d24.pt
# sample hparams
CFG=3.0
ITER=1
GUMBEL=0.01
# sample script
python sample.py \
 --model BiGR-L \
 --ckpt_bae $CKPT_BAE \
 --dataset custom --codebook_size $CODE --img_size 256 --norm_first \
 --ckpt $CKPT_BIGR \
 --num-classes 1000 \
 --seed 0 \
 --p_flip \
 --focal 0.0 \
 --use_adaLN \
 --cfg-scale $CFG \
 --num_sample_iter $ITER \
 --gumbel_temp $GUMBEL \
 --cfg_schedule constant \
 --save-path samples