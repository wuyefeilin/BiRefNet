#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
test_dir="test"
exp="sky1000"
model="ckpt/${exp}/epoch_150.pth"
output="ckpt/${exp}/${test_dir}_model_150_align_false"
python predict.py \
    --img_root "/root/cgw/SegRefine/data/sky_test/${test_dir}/image" \
    --ckpt $model \
    --output "output"
    
    # $output