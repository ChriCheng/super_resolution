export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
python eval.py \
  --set5_hr_dir ./datasets/DIV2K_train_HR \
  --ckpt_path ./outputs/train_espcn_x4/best.ckpt \
  --save_dir ./outputs/eval_set5_x4 \
  --device_target GPU
