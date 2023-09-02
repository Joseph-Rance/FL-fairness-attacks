#!/bin/bash
cd /nfs-share/jr897/FL-fairness-attacks
source ../miniconda3/bin/activate workspace
[ -d "outputs" ] || (echo "added directory 'outputs'" && mkdir outputs)
for MAL in 0 20
do
    for OPTIMISER in sgd adam
    do
        echo "running with $MAL malicious clients, and optimiser $OPTIMISER"
        printf "name: config_gen_${MAL}_${OPTIMISER}\nseed: 0\nclients:\n  num: 10\n  num_malicious: $MAL\n  fraction_fit: 1.0\ntraining:\n  optimiser: $OPTIMISER\n  batch_size: 1024\n  rounds: 50" > configs/config_gen.yaml
        python src/main.py config_gen.yaml > "outputs/out_${MAL}_${OPTIMISER}"
    done
done