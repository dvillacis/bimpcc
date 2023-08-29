#!/bin/bash

# Run the DTV experiment for the wood dataset
# Usage: ./run_dtv_experiment.sh datasets/wood

for sz in 1
do
    echo "Running TGV experiment for size $sz"
    python experiments/tgv_denoising.py datasets/wood tgv_denoising_results/0.1 --img_scale 0.1 --patch_size $sz 
done