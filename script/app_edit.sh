# model used
CODE=24
CKPT_BAE=ckpts/bae/bae_d24/binaryae_ema_1000000.th
CKPT_BIGR=ckpts/gpt/bigr_L_d24.pt
# editing info
IMG=app_data/editing/n01518878_ostrich.JPEG
CLS=296
# sample script
python apps/inpaint.py \
 --model BiGR-L \
 --ckpt_bae $CKPT_BAE \
 --dataset custom --codebook_size $CODE --img_size 256 --norm_first \
 --ckpt $CKPT_BIGR \
 --num-classes 1000 \
 --seed 0 \
 --focal 0.0 \
 --p_flip \
 --use_adaLN \
 --cfg-scale 2.5 \
 --temperature 1.0 \
 --num_sample_iter 25 \
 --input_image $IMG \
 --mode edit \
 --save-path samples \
 --cond_edit $CLS