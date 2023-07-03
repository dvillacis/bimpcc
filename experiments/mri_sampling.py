import argparse
import os
import numpy as np
from pathlib import Path
# from bimpcc.dataset import 
from bimpcc.dataset import SubsamplingDataset as Dataset
from bimpcc.operators import FirstDerivative, CircularPatchOperator
from bimpcc.mpcc import solve_mpcc

# Parse arguments
parser = argparse.ArgumentParser(description='MRI sampling.')
parser.add_argument('dataset_dir', type=str, help='Path to dataset directory')
parser.add_argument('output_dir', type=str, help='Path to output directory')
parser.add_argument('--subsampling', type=float, default=0.9, help='Subsampling of the k-space.')
parser.add_argument('--img_scale', type=float, default=0.1, help='Image scale.')
parser.add_argument('--patch_size', type=int, default=1, help='Patch size.')
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
output_dir = Path(args.output_dir)

if dataset_dir.exists() == False:
    raise Exception('Dataset directory does not exist.')
if output_dir.exists() == False:
    os.makedirs(output_dir)

dataset = Dataset(dataset_dir,args.img_scale)
true_img, f, Rop, Fop = dataset.get_data(subsampling=args.subsampling)
n,m = true_img.shape

# Define the required operators
Kx = FirstDerivative(n*m,dims=(n,m),dir=0)
Ky = FirstDerivative(n*m,dims=(n,m),dir=1)
R = Rop * Fop
u = int(np.sqrt(Kx.shape[0]))
Q = CircularPatchOperator((u,u),args.patch_size)

param,sol,q,r,delta,theta,extra = solve_mpcc(
    true_img=true_img,
    noisy_img=f,
    Kx=Kx,
    Ky=Ky,
    R=R,
    Q=Q,
    tik=0.2,
    alpha_size=args.patch_size,
    tol_max=1.0,
    tol_min=0.1
)


# Save results
results_dir = output_dir / f'mri_sampling_scale_{args.img_scale}_subsampling_{args.subsampling}_patch_{args.patch_size}.npy'
stats_dir = output_dir / f'mri_sampling_scale_{args.img_scale}_subsampling_{args.subsampling}_patch_{args.patch_size}_stats.npy'
data = {'param':param,'sol':sol,'true_img':true_img,'noisy_img':np.real(R.H*f)}
with open(results_dir,'wb') as f:
    np.save(f,data)
    print(f'Saved results to {results_dir}.')
with open(stats_dir,'wb') as f:
    np.save(f,extra)
    print(f'Saved results to {stats_dir}.')