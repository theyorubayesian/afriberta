#!/bin/bash
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:0
#SBATCH --gres=gpu:a100:1

# export CUDA_AVAILABLE_DEVICES=0
data_path=$1
vocab_size=$2

train_data_path=$data_path/train/all_train.txt
output_path=spm_model_new


mkdir -p $output_path

echo "Learning subword units..."
python src/subword.py --input=$train_data_path --model_prefix=$output_path/sentencepiece.bpe --vocab_size=$vocab_size
echo "Done! All files below saved in ${output_path}"
ls $output_path
