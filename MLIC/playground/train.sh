work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
nohup python train.py --metrics mse --exp mlicpp_on_ssp --gpu_id 0 --lambda 0.0018 -lr 1e-4 --clip_max_norm 1.0 --seed 42 --batch-size 32 --patch-size 196 256   #-c /data2/jiangwei/work_space/MLICPP/playground/experiments/mlicpp_mse_q1/checkpoints/checkpoint_025.pth.tar
