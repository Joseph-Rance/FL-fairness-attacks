#!/bin/bash
cd /nfs-share/jr897/FL-fairness-attacks
source ../miniconda3/bin/activate workspace
[ -d "outputs" ] || (echo "added directory 'outputs'" && mkdir outputs)
for ROUND in 0 500
do
    for OPTIMISER in sgd adam
    do
        echo "running with $MAL malicious clients, and optimiser $OPTIMISER"
        printf "name: config_gen_${MAL}_${OPTIMISER}\nseed: 0\nclients:\n  num: 5\n  num_malicious: 20\n  attack_round: $ROUND\n  fraction_fit: 1.0\ntraining:\n  optimiser: $OPTIMISER\n  batch_size: 2048\n  rounds: 50" > configs/config_gen.yaml
        python src/main.py config_gen.yaml > "outputs/out_${MAL}_${OPTIMISER}"
    done
done