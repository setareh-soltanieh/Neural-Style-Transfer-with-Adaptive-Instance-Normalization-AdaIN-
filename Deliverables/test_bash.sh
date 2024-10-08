#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH --account=ingenuitylabs
#SBATCH --partition=Aurora
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=test.log
#SBATCH --error=test.err

nvidia-smi
python3 test.py -content_image images/content/baboon.jpg -style_image images/style/brushstrokes.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.9 -cuda Y
python3 test.py -content_image images/content/baboon.jpg -style_image images/style/0a585acb9d7134c0b39656a588527385c.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.9 -cuda Y
python3 test.py -content_image images/content/baboon.jpg -style_image images/style/Andy_Warhol_97.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.9 -cuda Y
python3 test.py -content_image images/content/baboon.jpg -style_image images/style/the-persistence-of-memory-1931.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.9 -cuda Y
python3 test.py -content_image images/content/baboon.jpg -style_image images/style/chagall_marc_1.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.9 -cuda Y