#!/bin/bash

# Run the DTV experiment for the wood dataset
# Usage: ./run_dtv_experiment.sh datasets/wood

for sz in 1
do
    echo "Running TV experiment for patch size $sz"
    python experiments/tv_denoising.py datasets/wood tv_denoising_results/0.3 --img_scale 0.3 --patch_size $sz 
done