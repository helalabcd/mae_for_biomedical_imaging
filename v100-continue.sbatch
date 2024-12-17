#! /bin/bash
#SBATCH -c 6
#SBATCH --mem 32G
#SBATCH -p grete-h100
#SBATCH -t 2880
#SBATCH -G H100:1
mamba init
source /home/nimdalin/.bashrc
mamba activate facebookmae
python main_pretrain.py --data_path=../../mae_for_biomedical_imaging/HeLa_dataset/ --sequence_length=1 --batch_size=16 --warmup_epochs=1 --resume=output_dir/checkpoint-31.pth --start_epoch=31

