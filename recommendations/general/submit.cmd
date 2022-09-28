#!/bin/bash -l
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p gpunodes
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=16384

module load cuda
source /u/striethk/.bashrc

source activate molar
python prepare_gryffin.py
conda activate gryffin-tf
python run_gryffin.py
conda activate molar
python process_gryffin.py
