export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

python train.py \
  --div2k_hr_dir ./data/DIV2K_train_HR \
  --save_dir ./outputs/train_espcn_x4 \
  --epochs 80 \
  --batch_size 16 \
  --patch_size 192 \
  --repeat 20 \
  --lr 1e-3 \
  --device_target CPU
