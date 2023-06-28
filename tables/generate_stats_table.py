import argparse
from pathlib import Path
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description='Plot results.')
parser.add_argument('results_dir', type=str, help='Path to results directory')
parser.add_argument('output_dir', type=str, help='Path to output directory')
args = parser.parse_args()

results_dir = Path(args.results_dir)
output_dir = Path(args.output_dir)
if results_dir.exists() == False:
    raise Exception('Results directory does not exist.')
if output_dir.exists() == False:
    raise Exception('Output directory does not exist.')

# Load results
result_data = np.load(results_dir,allow_pickle=True)
print(result_data)