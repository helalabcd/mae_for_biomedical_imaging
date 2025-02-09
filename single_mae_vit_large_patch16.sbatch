#! /bin/bash
#SBATCH -c 6
#SBATCH --mem 32G
#SBATCH -p grete-h100:shared
#SBATCH -t 2880
#SBATCH -G H100:1
source ~/.bashrc
conda init
conda activate facebookmae
python main_pretrain.py --data_path=/mnt/lustre-grete/usr/u14148/HeLa_dataset/ --sequence_length=1 --batch_size=16 --warmup_epochs=1 --model=mae_vit_large_patch16
