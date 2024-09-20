#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH --account=ingenuitylabs
#SBATCH --partition=Aurora
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=train_1k.log
#SBATCH --error=train_1k.err

nvidia-smi
python3 my_train_1k.py -content_dir Datasets/COCO1k/ -style_dir Datasets/wikiart1k/ -gamma 1.0 -e 1000 -b 20 -l encoder.pth -s decoder.pth -p decoder.png -cuda Y -save_model_interval 10 -save_weights model_1k