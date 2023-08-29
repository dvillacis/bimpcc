import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pylops import Identity, VStack
from bimpcc.dataset import NoiseDataset as Dataset
from bimpcc.operators import FirstDerivative, PatchOperator
from bimpcc.tv_two_dim import solve_mpcc
from pyproximal import L2,L21
from pyproximal.optimization.primaldual import PrimalDual

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
A = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
true_img = np.kron(A,np.ones((5,5)))
noise = np.random.normal(0,0.1,true_img.shape)
noisy_img = true_img + noise
# print(true_img,true_img.shape)


# plt.plot(rangex,true_img)
# plt.plot(rangex,noisy_img)
# plt.show()

n,m = true_img.shape

# Define the required operators
Kx = FirstDerivative(n*m,dims=(n,m),dir=0)
Ky = FirstDerivative(n*m,dims=(n,m),dir=1)
K = VStack([Kx,Ky])
R = Identity(n*m)
Q = PatchOperator((n-1,m-1),(args.patch_size,args.patch_size))

# Define the MPCC model
sol,extra = solve_mpcc(
    true_img=true_img,
    noisy_img=noisy_img,
    K=K,
    R=R,
    Q=Q,
    α_size=args.patch_size,
    print_level=5
)

# print(param)

# Solution using linearized ADMM
l2 = L2(b=noisy_img.ravel())
l1iso = L21(ndim=2, sigma=sol.xStar['α'][0])

L = np.real((K.H * K).eigs(neigs=1, which='LM')[0])
tau = 1.
mu = 0.99 * tau / L
xladmm = PrimalDual(l2, l1iso, K, tau=tau, mu=mu,x0=noisy_img.ravel(), niter=300,show=True)

α = sol.xStar['α'][0]
u = sol.xStar['u'].reshape(true_img.shape)
xladmm = xladmm.reshape(true_img.shape)
diff = np.abs(u-true_img)

ax,fig = plt.subplots(1,5)
fig[0].imshow(true_img,cmap='gray')
fig[0].set_title('True Image')
fig[0].set_xticklabels([])
fig[0].set_yticklabels([])
fig[0].set_xticks([])
fig[0].set_yticks([])
fig[1].imshow(noisy_img,cmap='gray')
fig[1].set_title('Noisy Image')
fig[1].set_xticklabels([])
fig[1].set_yticklabels([])
fig[1].set_xticks([])
fig[1].set_yticks([])
fig[2].imshow(u,cmap='gray')
fig[2].set_title(f'MPCC - {α:.3f}')
fig[2].set_xticklabels([])
fig[2].set_yticklabels([])
fig[2].set_xticks([])
fig[2].set_yticks([])
fig[3].imshow(xladmm,cmap='gray')
fig[3].set_title(f'LADMM - {α:.3f}')
fig[3].set_xticklabels([])
fig[3].set_yticklabels([])
fig[3].set_xticks([])
fig[3].set_yticks([])
fig[4].imshow(diff,cmap='gray')
fig[4].set_title(f'diff - {np.linalg.norm(diff.ravel()):.3f}')
fig[4].set_xticklabels([])
fig[4].set_yticklabels([])
fig[4].set_xticks([])
fig[4].set_yticks([])
plt.show()

# plt.plot(rangex,true_img,label='True Signal')
# # plt.plot(rangex,noisy_img)
# plt.plot(rangex,sol,'--',label='MPCC')
# plt.plot(rangex,xladmm,'-o',label='LADMM')
# # plt.plot(rangex[:-1],extra.xStar['c'],'--',label='c')
# plt.legend()
# plt.grid()
# plt.show()

# # Save results
# results_dir = output_dir / f'tv_denoising_scale_{args.img_scale}.npy'
# data = {'param':param,'sol':sol,'true_img':true_img,'noisy_img':noisy_img}
# with open(results_dir,'wb') as f:
#     np.save(f,data)
#     print(f'Saved results to {results_dir}.')
