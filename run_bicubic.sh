source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate sr39

export CUDA_HOME=/usr/local/cuda-11.6
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

cd /root/code/super_resolution || exit 1

python eval_bicubic.py \
  --eval_hr_dir ./data/DIV2K_valid_HR \
  --save_dir ./outputs/eval_bicubic_div2k_valid_x4 \
  --device_target GPU