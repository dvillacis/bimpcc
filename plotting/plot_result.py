import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from bimpcc.operators import DiagonalPatchOperator, DirectionalGradient_Fixed
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import exposure

# Parse arguments
parser = argparse.ArgumentParser(description='Plot results.')
parser.add_argument('results_dir', type=str, help='Path to results directory')
parser.add_argument('output_dir', type=str, help='Path to output directory')
parser.add_argument('--show', type=bool, default=True, help='Show plots.')
args = parser.parse_args()

results_dir = Path(args.results_dir)
output_dir = Path(args.output_dir)
if results_dir.exists() == False:
    raise Exception('Results directory does not exist.')
if output_dir.exists() == False:
    raise Exception('Output directory does not exist.')

# Load results
result_data = np.load(results_dir,allow_pickle=True)
true_img = result_data.item()['true_img']
noisy_img = result_data.item()['noisy_img'].reshape(true_img.shape)
sol = result_data.item()['sol'].reshape(true_img.shape)
param = result_data.item()['param']

recons = exposure.match_histograms(sol,true_img)

l = int(np.sqrt(len(param)))
m,n = true_img.shape
Kx = DirectionalGradient_Fixed((m,n),3,-0.61,dir=0)
Ky = DirectionalGradient_Fixed((m,n),3,-0.61,dir=1)
u = int(np.sqrt(Kx.shape[0]))
Q = DiagonalPatchOperator((u,u),(l,l))
print(Q)

# plot results
if len(param) > 1:
    fig,ax = plt.subplots(1,4,figsize=(15,5))
else:
    fig,ax = plt.subplots(1,3,figsize=(15,5))

ax[0].imshow(true_img,cmap='gray')
ax[0].set_title('True image')
ax[0].set_xticklabels([])
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_yticklabels([])
ax[1].imshow(noisy_img,cmap='gray')
ax[1].set_title('Data')
ax[1].set_xticklabels([])
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_yticklabels([])
ax[1].set_xlabel(f'PSNR={psnr(true_img,noisy_img):.4f}\nSSIM={ssim(true_img,noisy_img,data_range=noisy_img.max() - noisy_img.min()):.4f}')
ax[2].imshow(recons,cmap='gray')
ax[2].set_title('Reconstruction')
ax[2].set_xticklabels([])
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_yticklabels([])
ax[2].set_xlabel(f'PSNR={psnr(true_img,recons):.4f}\nSSIM={ssim(true_img,recons,data_range=true_img.max() - true_img.min()):.4f}')
if len(param) > 1:
    p = Q*param
    v = int(np.sqrt(len(p)))
    par = ax[3].imshow(p.reshape((v,v)),cmap='gray')
    ax[3].set_title('Learned Parameter')
    ax[3].set_xticklabels([])
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_yticklabels([])
    cb = plt.colorbar(par,ax=ax[3],orientation='vertical')

plt.show()


