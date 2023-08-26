#!/bin/bash
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --job-name=fl_fair_att

cd /nfs-share/jr897/FL-fairness-attacks
source ../miniconda3/bin/activate workspace
srun sh run.sh