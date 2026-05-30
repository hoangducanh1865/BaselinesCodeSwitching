#!/bin/bash -l
#SBATCH --job-name=clear_asr
#SBATCH --partition=defq
#SBATCH --output=/home/user14/anhhd/asr/BaselinesCodeSwitching/baselines/CLEAR/logs/train_%j.out
#SBATCH --error=/home/user14/anhhd/asr/BaselinesCodeSwitching/baselines/CLEAR/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10g
#SBATCH --gres=gpu:a100:1

cd /home/user14/anhhd/asr/BaselinesCodeSwitching/baselines/CLEAR

source /home/user14/miniconda3/bin/activate CLEAR

mkdir -p logs

echo "Bắt đầu train CLEAR (Whisper-small, freeze encoder, descriptive prompting)..."
python whisper_fine.py \
  --dataset mucs \
  --model small \
  --freeze \
  --exp-name clear_small \
  --batch 16 \
  --epoch 10 \
  --lr 1e-4
echo "Hoàn thành!"