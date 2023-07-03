import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Parse arguments
parser = argparse.ArgumentParser(description='Plot results.')
parser.add_argument('results_dirs', type=str, nargs='+', help='Path to results directory')
parser.add_argument('output_dir', type=str, help='Path to output directory')
args = parser.parse_args()

if len(args.results_dirs) == 0:
    raise Exception('There are no results dirs')

output_dir = Path(args.output_dir)
if output_dir.exists() == False:
        raise Exception('Output directory does not exist.')

table = pd.DataFrame(columns=['Num Patch','Phase','fopt','niter','ngev','time'])
for j,results_dir in enumerate(args.results_dirs):
    print(results_dir)
    results_dir = Path(results_dir)
    
    if results_dir.exists() == False:
        raise Exception('Results directory does not exist.')
    
    # Load results
    
    result_data = np.load(results_dir,allow_pickle=True)
    for i,d in enumerate(result_data):
        data = {
            'Num Patch':j+1,
            'Phase':i+1,
            'fopt':d['fStar'],
            'niter':d['info']['userObjCalls'],
            'ngev':d['info']['userSensCalls'],
            'time':d['info']['optTime']
        }
        table = pd.concat([table,pd.DataFrame(data,index=[i])])
table = table.set_index(['Num Patch','Phase'])
# Swap levels of the MultiIndex column index
# print(table)
# table = table.swaplevel(0,1)

# print(table)
latex_table = table.to_latex()
print(latex_table)