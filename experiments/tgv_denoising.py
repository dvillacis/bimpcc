import argparse
import numpy as np
from pathlib import Path
from pylops import Identity, VStack, Block, Zero
from bimpcc.dataset import NoiseDataset as Dataset
from bimpcc.operators import FirstDerivative, PatchOperator
from bimpcc.tgv_two_dim import solve_mpcc

# Parse arguments
parser = argparse.ArgumentParser(description='Obtain optimal parameters of a TGV denoising model.')
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
    raise Exception('Output directory does not exist.')
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
true_img,noisy_img = dataset.get_data()
n,m = true_img.shape

# Define the required operators
Kx = FirstDerivative(n*m,dims=(n,m),dir=0)
Ky = FirstDerivative(n*m,dims=(n,m),dir=1)
K = VStack([Kx,Ky])
Z = Zero(Kx.shape[0],Kx.shape[1])

# Simmetrized gradien tensor 
Kx2 = FirstDerivative((n-1)*(m-1),dims=(n-1,m-1),dir=0)
Ky2 = FirstDerivative((n-1)*(m-1),dims=(n-1,m-1),dir=1)
Z2 = Zero(Kx2.shape[0],Kx2.shape[1])
E = Block([[Kx2,Z2],[np.sqrt(2)*Ky2,Z2],[Z2,Ky2]])
R = Identity(n*m)
u = int(np.sqrt(Kx.shape[0]))
Q = PatchOperator((u,u),(args.patch_size,args.patch_size))
S = PatchOperator((u-1,u-1),(args.patch_size,args.patch_size))

sol,extra = solve_mpcc(
    true_img=true_img,
    noisy_img=noisy_img,
    K=K,
    E=E,
    R=R,
    Q=Q,
    S=S,
    α_size=args.patch_size,
    β_size=args.patch_size,
    print_level=5
)

# Save results
results_dir = output_dir / f'tgv_denoising_scale_{args.img_scale}_patch_{args.patch_size}.npy'
stats_dir = output_dir / f'tgv_denoising_scale_{args.img_scale}_patch_{args.patch_size}_stats.npy'
data = {'param1':sol.xStar['α'],'param2':sol.xStar['β'],'sol':sol.xStar['v'],'true_img':true_img,'noisy_img':R.T*noisy_img.ravel()}
with open(results_dir,'wb') as f:
    np.save(f,data)
    print(f'Saved results to {results_dir}.')
with open(stats_dir,'wb') as f:
    np.save(f,extra)
    print(f'Saved results to {stats_dir}.')

