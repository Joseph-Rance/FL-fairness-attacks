#!/bin/bash
[ -d "outputs" ] || (echo "added directory 'outputs'" && mkdir outputs)
for ROUND in 0 10 500
do
    for OPTIMISER in sgd adam
    do
        for SEED in 0 1 2 3 4
        do
            echo "running with malicious clients starting on round $ROUND, seed $SEED, and optimiser $OPTIMISER"
            printf "name: config_gen_${ROUND}_${SEED}_${OPTIMISER}\nseed: $SEED\nclients:\n  num: 200\n  num_malicious: 20\n  attack_round: $ROUND\n  fraction_fit: 0.05\ntraining:\n  optimiser: $OPTIMISER\n  batch_size: 256\n  rounds: 20" > configs/config_gen.yaml
            python src/main.py config_gen.yaml > "outputs/out_${ROUND}_${SEED}_${OPTIMISER}"
        done
    done
done