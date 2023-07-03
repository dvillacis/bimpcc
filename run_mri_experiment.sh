#!/bin/bash

# Run the DTV experiment for the wood dataset
# Usage: ./run_dtv_experiment.sh datasets/wood

for sz in 1 2 4 6 8
do
    echo "Running MRI experiment for size $sz"
    python experiments/mri_sampling.py datasets/brain mri_sampling_results/0.2 --img_scale 0.2 --patch_size $sz --subsampling 0.5
done