#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH --account=ingenuitylabs
#SBATCH --partition=Aurora
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=train_10k.log
#SBATCH --error=train_10k.err

nvidia-smi
python3 my_train_10k.py -content_dir Datasets/COCO10k/COCO10k/ -style_dir Datasets/wikiart10k/wikiart10k/ -gamma 1.0 -e 1000 -b 20 -l encoder.pth -s decoder.pth -p decoder.png -cuda Y -save_model_interval 10 -save_weights model_10k -save_model_interval 10 -save_weights model_10k