#!/bin/bash
#SBATCH -p gpu22
#SBATCH -c 2
#SBATCH -o jobs/slurm-out%j.out
#SBATCH --gres gpu:a40:1
#SBATCH -t 04:00:00
#SBATCH --mem=32G

cd "/CT/Hands2/work/NNTI-project"
source "env_nnti/bin/activate"
python -u "scripts/task3.py"