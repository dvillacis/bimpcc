#!/bin/bash

# Run the DTV experiment for the wood dataset
# Usage: ./run_dtv_experiment.sh datasets/wood

for sz in 1 2 4 8 16
do
    echo "Running DTV experiment for size $sz"
    python experiments/dtv_inpainting.py datasets/wood dtv_inpainting_results/0.2 --img_scale 0.2 --patch_size $sz 
done