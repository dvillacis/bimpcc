import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pylops import Identity
from bimpcc.dataset import NoiseDataset as Dataset
from bimpcc.operators import FirstDerivative, PatchOperator
from bimpcc.tv_one_dim import solve_mpcc
from pyproximal import L1,L2
from pyproximal.optimization.primal import LinearizedADMM

# Parse arguments
parser = argparse.ArgumentParser(description='Obtain optimal parameters of a TV denoising model.')
parser.add_argument('--tik', type=float, default=1e-3, help='Tikhonov regularization parameter.')
parser.add_argument('--maxiter', type=int, default=100, help='Maximum number of iterations.')
parser.add_argument('--tol', type=float, default=1e-3, help='Tolerance.')
parser.add_argument('--print_sparsity', type=bool, default=False, help='Print sparsity pattern.')
parser.add_argument('--patch_size', type=int, default=1, help='Patch size.')
parser.add_argument('--img_scale', type=float, default=0.1, help='Image scale.')
args = parser.parse_args()

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
np.random.seed(1234)
rangex = np.linspace(0,1.5,100)
true_img = np.piecewise(rangex,[rangex < 0.1,(rangex >= 0.1) & (rangex < 0.75),(rangex >= 0.75) & (rangex < 0.95),(rangex >= 0.95) & (rangex < 1.2),rangex >= 1.2],(0.2,lambda x: x,0.3,1,0.75))
noisy_img = true_img + np.random.normal(0,0.05,true_img.shape)

# plt.plot(rangex,true_img)
# plt.plot(rangex,noisy_img)
# plt.show()

n = len(true_img)

# Define the required operators
K = FirstDerivative(n,dir=0)
R = Identity(n)
Q = PatchOperator((n-1,1),(args.patch_size,1))

# Define the MPCC model
param,sol,extra = solve_mpcc(
    true_sgn=true_img,
    noisy_sgn=noisy_img,
    K=K,
    R=R,
    Q=Q,
    α_size=args.patch_size,
    print_level=5
)

print(param)

# Solution using linearized ADMM
l2 = L2(b=noisy_img)
l1 = L1(sigma=param)

L = np.real((K.H * K).eigs(neigs=1, which='LM')[0])
tau = 1.
mu = 0.99 * tau / L
xladmm, _ = LinearizedADMM(l2, l1, K, tau=tau, mu=mu,x0=noisy_img, niter=200,show=True)


plt.plot(rangex,true_img,label='True Signal')
# plt.plot(rangex,noisy_img)
plt.plot(rangex,sol,'--',label='MPCC')
plt.plot(rangex,xladmm,'-.',label='LADMM')
# plt.plot(rangex[:-1],extra.xStar['c'],'--',label='c')
plt.legend()
plt.grid()
plt.show()

# # Save results
# results_dir = output_dir / f'tv_denoising_scale_{args.img_scale}.npy'
# data = {'param':param,'sol':sol,'true_img':true_img,'noisy_img':noisy_img}
# with open(results_dir,'wb') as f:
#     np.save(f,data)
#     print(f'Saved results to {results_dir}.')
