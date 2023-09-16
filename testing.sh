#!/bin/bash
#SBATCH --job-name=fairness_attacks
#SBATCH -c 4
#SBATCH --gres=gpu:1
cd /nfs-share/jr897/FL-fairness-attacks
source ../miniconda3/bin/activate workspace
[ -d "outputs" ] || (echo "added directory 'outputs'" && mkdir outputs)
for ROUND in 0
do
    for OPTIMISER in sgd
    do
        echo "running malicious on round $ROUND, and optimiser $OPTIMISER"
        printf "name: config_gen_${ROUND}_${OPTIMISER}\nseed: 0\nclients:\n  num: 2\n  num_malicious: 2\n  attack_round: $ROUND\n  fraction_fit: 1.0\ntraining:\n  optimiser: $OPTIMISER\n  batch_size: 2048\n  rounds: 50" > configs/config_gen.yaml
        python src/main.py config_gen.yaml
    done
done
