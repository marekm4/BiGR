# model used
CODE=24
CKPT_BAE=ckpts/bae/bae_d24/binaryae_ema_1000000.th
CKPT_BIGR=ckpts/gpt/bigr_L_d24.pt
# interpolating info
CLS1=284
CLS2=388
# sample script
python apps/interpolate.py \
 --model BiGR-L \
 --ckpt_bae $CKPT_BAE \
 --dataset custom --codebook_size $CODE --img_size 256 --norm_first \
 --ckpt $CKPT_BIGR \
 --num-classes 1000 \
 --cfg-scale 2.5 \
 --temperature 1.0 \
 --seed 0 \
 --focal 0.0 \
 --p_flip \
 --use_adaLN \
 --num_sample_iter 20 \
 --cls1 $CLS1 \
 --cls2 $CLS2 \
 --save-path samples
