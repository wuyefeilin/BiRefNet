#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# test_dir="test"
test_dir="jira_1107"
exp="sky1000"
model="ckpt/${exp}/epoch_150.pth"
output="ckpt/${exp}/${test_dir}_model_150"
python predict.py \
    --img_root "/root/cgw/SegRefine/data/sky_test/${test_dir}" \
    --ckpt $model \
    --output $output