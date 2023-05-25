import argparse
from pathlib import Path
from pylops import Identity
from bimpcc.dataset import Dataset
from bimpcc.operators import FirstDerivative
from bimpcc.mpcc import MPCC

# Parse arguments
parser = argparse.ArgumentParser(description='Obtain optimal parameters of a TV denoising model.')
parser.add_argument('--dataset', type=str, default='wood', help='Dataset to use.')
parser.add_argument('--output', type=str, default='output', help='Output directory.')
parser.add_argument('--tik', type=float, default=1e-3, help='Tikhonov regularization parameter.')
parser.add_argument('--maxiter', type=int, default=100, help='Maximum number of iterations.')
parser.add_argument('--tol', type=float, default=1e-3, help='Tolerance.')
parser.add_argument('--print_sparsity', type=bool, default=False, help='Print sparsity pattern.')
parser.add_argument('--patch_size', type=int, default=1, help='Patch size.')
parser.add_argument('--img_scale', type=float, default=1.0, help='Image scale.')
args = parser.parse_args()

dataset_dir = Path(args.dataset)
output_dir = Path(args.output)
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
R = Identity(n*m)

# Define the MPCC model
mpcc = MPCC(true_img=true_img,noisy_img=noisy_img,Kx=Kx,Ky=Ky,R=R,alpha_size=args.patch_size)
param,sol = mpcc.solve()