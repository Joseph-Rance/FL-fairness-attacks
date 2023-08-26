#!/bin/bash
[ -d "outputs" ] || (echo "added directory 'outputs'" && mkdir outputs)
for MAL in 0 20
do
    for SEED in 0 1 2 3 4
    do
        for OPTIMISER in adam sgd
        do
            echo "running with $MAL malicious clients, seed $SEED, and optimiser $OPTIMISER"
            printf "name: config_gen_${MAL}_${SEED}_${OPTIMISER}\nseed: $SEED\nclients:\n  num: 200\n  num_malicious: $MAL\n  fraction_fit: 0.05\ntraining:\n  optimiser: $OPTIMISER\n  batch_size: 256\n  rounds: 20" > configs/config_gen.yaml
            python src/main.py config_gen.yaml > "outputs/out_${MAL}_${SEED}_${OPTIMISER}"
        done
    done
done