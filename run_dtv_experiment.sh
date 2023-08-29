#!/bin/bash

# Run the DTV experiment for the wood dataset
# Usage: ./run_dtv_experiment.sh datasets/wood

for sz in 1
do
    echo "Running DTV experiment for patch size $sz"
    python experiments/dtv_inpainting.py datasets/wood dtv_inpainting_results/0.1 --img_scale 0.1 --patch_size $sz 
done