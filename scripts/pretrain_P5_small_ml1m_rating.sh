# Run with $ bash scripts/pretrain_P5_small_ml1m_rating.sh 4
# Rating prediction only for ML-1M dataset

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

name=ml1m-small-rating

output=snap/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 12350 \
    src/pretrain.py \
        --distributed --multiGPU \
        --seed 2022 \
        --train ml-1m \
        --valid ml-1m \
        --batch_size 32 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 1e-3 \
        --num_workers 4 \
        --clip_grad_norm 1.0 \
        --losses 'rating' \
        --backbone 't5-small' \
        --output $output ${@:2} \
        --epoch 10 \
        --max_text_length 512 \
        --gen_max_length 64 \
        --whole_word_embed > $name.log
