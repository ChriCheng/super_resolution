#!/bin/bash
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate sr39

export CUDA_HOME=/usr/local/cuda-11.6
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

cd /root/code/super_resolution || exit 1

exec python SwinIR_train.py \
  --div2k_hr_dir ./data/DIV2K_train_HR \
  --div2k_val_hr_dir ./data/DIV2K_valid_HR \
  --save_dir ./outputs/train_swinir_x4 \
  --epochs 30 \
  --batch_size 16 \
  --patch_size 192 \
  --repeat 5 \
  --lr 2e-4 \
  --device_target GPU