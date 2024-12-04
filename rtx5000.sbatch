#! /bin/bash
#SBATCH -c 6
#SBATCH --mem 32G
#SBATCH -p gpu
#SBATCH -t 7200
#SBATCH -G V100:1
source ~/.bashrc
mamba activate facebookmae
python main_pretrain.py --data_path=../../datasets/HeLa_dataset/ --sequence_length=4 --batch_size=16 --warmup_epochs=1
