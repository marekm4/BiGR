# model used
CODE=24
CKPT_BAE=ckpts/bae/bae_d24/binaryae_ema_1000000.th
CKPT_BIGR=ckpts/gpt/bigr_L_d24.pt
# enriching info
IMG=app_data/enrichment/n01514668_cock.JPEG
# sample script
python apps/enrich.py \
 --model BiGR-L \
 --ckpt_bae $CKPT_BAE \
 --dataset custom --codebook_size $CODE --img_size 256 --norm_first \
 --ckpt $CKPT_BIGR \
 --num-classes 1000 \
 --cfg-scale 10.0 \
 --temperature 1.0 \
 --seed 1 \
 --focal 0.0 \
 --p_flip \
 --use_adaLN \
 --num_sample_iter 20 \
 --downsample_rate 2 \
 --input_image $IMG \
 --save-path samples