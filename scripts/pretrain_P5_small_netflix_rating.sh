#!/bin/bash
# Run with $ bash scripts/pretrain_P5_small_netflix_rating.sh
# Rating prediction only for Netflix dataset (single GPU mode)

export CUDA_VISIBLE_DEVICES=0

name=netflix-small-rating

output=snap/$name

PYTHONPATH=$PYTHONPATH:./src \
python src/pretrain.py \
    --seed 2022 \
    --train netflix \
    --valid netflix \
    --batch_size 16 \
    --optim adamw \
    --warmup_ratio 0.05 \
    --lr 1e-3 \
    --num_workers 4 \
    --clip_grad_norm 1.0 \
    --losses 'rating' \
    --backbone 't5-small' \
    --output $output $@ \
    --epoch 10 \
    --max_text_length 512 \
    --gen_max_length 64 \
    --whole_word_embed 2>&1 | tee $name.log
