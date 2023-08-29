import argparse
import os
import numpy as np
from pathlib import Path
from pylops import Identity, VStack
from bimpcc.dataset import InpaintingDataset as Dataset
from bimpcc.operators import DirectionalGradient_Fixed, DiagonalPatchOperator
from bimpcc.tv_mpcc_2 import solve_mpcc, solve_mpcc_one_shot

# Parse arguments
parser = argparse.ArgumentParser(description='Obtain optimal parameters of a TV denoising model.')
parser.add_argument('dataset_dir', type=str, help='Path to dataset directory')
parser.add_argument('output_dir', type=str, help='Path to output directory')
parser.add_argument('--tik', type=float, default=1e-3, help='Tikhonov regularization parameter.')
parser.add_argument('--maxiter', type=int, default=100, help='Maximum number of iterations.')
parser.add_argument('--tol', type=float, default=1e-3, help='Tolerance.')
parser.add_argument('--print_sparsity', type=bool, default=False, help='Print sparsity pattern.')
parser.add_argument('--patch_size', type=int, default=1, help='Patch size.')
parser.add_argument('--img_scale', type=float, default=0.1, help='Image scale.')
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
output_dir = Path(args.output_dir)
if dataset_dir.exists() == False:
    raise Exception('Dataset directory does not exist.')
if output_dir.exists() == False:
    os.makedirs(output_dir)
if args.tik < 0:
    raise Exception('Tikhonov regularization parameter must be nonnegative.')
if args.maxiter < 0:
    raise Exception('Maximum number of iterations must be nonnegative.')
if args.tol < 0:
    raise Exception('Tolerance must be nonnegative.')
if args.patch_size < 1:
    raise Exception('Patch size must be positive.')
if args.patch_size < 1:
    raise Exception('Patch size must be positive.')

# Load dataset
dataset = Dataset(dataset_dir, args.img_scale)
true_img,noisy_img,R = dataset.get_data()
n,m = true_img.shape

# Define the required operators
Kx = DirectionalGradient_Fixed((m,n),3,-0.61,dir=0)
Ky = DirectionalGradient_Fixed((m,n),3,-0.61,dir=1)
K = VStack([Kx,Ky])
Q = DiagonalPatchOperator((n-1,m-1),args.patch_size)

# Define the MPCC model
π_0 = np.linalg.norm(true_img.ravel()-(R.T@noisy_img).ravel())
print(f'π0 = {π_0}')
param,sol,q,r,delta,theta,extra = solve_mpcc(
    true_img=true_img,
    noisy_img=noisy_img,
    K=K,
    R=R,
    Q=Q,
    α_size=args.patch_size,
    tik=0.1,
    π_init=π_0,
    α_max=1.0,
)

# Save results
results_dir = output_dir / f'dtv_denoising_scale_{args.img_scale}_patch_{args.patch_size}.npy'
stats_dir = output_dir / f'dtv_denoising_scale_{args.img_scale}_patch_{args.patch_size}_stats.npy'
data = {'param':param,'sol':sol,'true_img':true_img,'noisy_img':R.T*noisy_img.ravel()}
with open(results_dir,'wb') as f:
    np.save(f,data)
    print(f'Saved results to {results_dir}.')
with open(stats_dir,'wb') as f:
    np.save(f,extra)
    print(f'Saved results to {stats_dir}.')
