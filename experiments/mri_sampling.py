import argparse
import numpy as np
from pathlib import Path
from bimpcc.dataset import load_shepp_logan_phantom
from bimpcc.operators import FirstDerivative
from bimpcc.mpcc import solve_mpcc

# Parse arguments
parser = argparse.ArgumentParser(description='MRI sampling.')
parser.add_argument('output_dir', type=str, help='Path to output directory')
parser.add_argument('--subsampling', type=float, default=0.7, help='Subsampling of the k-space.')
parser.add_argument('--img_scale', type=float, default=0.1, help='Image scale.')
parser.add_argument('--patch_size', type=int, default=1, help='Patch size.')
args = parser.parse_args()

output_dir = Path(args.output_dir)
if output_dir.exists() == False:
    raise Exception('Output directory does not exist.')

true_img, f, Rop, Fop = load_shepp_logan_phantom(scale=args.img_scale,subsampling=args.subsampling)
n,m = true_img.shape

# Define the required operators
Kx = FirstDerivative(n*m,dims=(n,m),dir=0)
Ky = FirstDerivative(n*m,dims=(n,m),dir=1)
R = Rop * Fop


param,sol,q,r,delta,theta = solve_mpcc(
    true_img=true_img,
    noisy_img=f,
    Kx=Kx,
    Ky=Ky,
    R=R,
    tik=0.1
)


# Save results
results_dir = output_dir / f'mri_sampling_scale_{args.img_scale}_subsampling_{args.subsampling}.npy'
data = {'param':param,'sol':sol,'true_img':true_img,'noisy_img':np.real(R.H*f)}
with open(results_dir,'wb') as f:
    np.save(f,data)
    print(f'Saved results to {results_dir}.')