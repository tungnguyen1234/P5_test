# Run with $ bash scripts/pretrain_P5_small_ml20m.sh 4

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

name=ml20m-small

output=snap/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 12321 \
    src/pretrain.py \
        --distributed --multiGPU \
        --seed 2022 \
        --train ml-20m \
        --valid ml-20m \
        --batch_size 16 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 1e-3 \
        --num_workers 4 \
        --clip_grad_norm 1.0 \
        --losses 'rating,sequential,traditional' \
        --backbone 't5-small' \
        --output $output ${@:2} \
        --epoch 10 \
        --max_text_length 512 \
        --gen_max_length 64 \
        --whole_word_embed > $name.log
