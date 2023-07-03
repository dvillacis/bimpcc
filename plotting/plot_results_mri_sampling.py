import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import exposure
import tikzplotlib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from bimpcc.operators import CircularPatchOperator, DirectionalGradient_Fixed

# Parse arguments
parser = argparse.ArgumentParser(description='Plot group of results')
parser.add_argument('results_dirs', type=str, nargs='+', help='Result directories')
parser.add_argument('output_dir', type=str, help='Path to output directory')
args = parser.parse_args()

save = False

print(args)

if len(args.results_dirs) == 0:
    raise Exception('There are no results dirs')

fig,ax = plt.subplots(2,len(args.results_dirs)+2,figsize=(12,6))

# Load results
init_data = np.load(args.results_dirs[0],allow_pickle=True)
true_img = init_data.item()['true_img']
noisy_img = init_data.item()['noisy_img'].reshape(true_img.shape)
ax[0,0].imshow(true_img,cmap='gray')
ax[0,0].set_title('Original')
ax[0,0].set_xticklabels([])
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
ax[0,0].set_yticklabels([])
ax[0,1].imshow(noisy_img,cmap='gray')
ax[0,1].set_title('Damaged')
ax[0,1].set_xticklabels([])
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])
ax[0,1].set_yticklabels([])
ax[0,1].set_xlabel(f'PSNR={psnr(true_img,noisy_img):.4f}\nSSIM={ssim(true_img,noisy_img,data_range=noisy_img.max() - noisy_img.min()):.4f}')

ax[1,0].set_xticklabels([])
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])
ax[1,0].set_yticklabels([])
ax[1,0].axis('off')
ax[1,1].set_xticklabels([])
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])
ax[1,1].set_yticklabels([])
ax[1,1].axis('off')

m,n = true_img.shape
Kx = DirectionalGradient_Fixed((m,n),3,-0.61,dir=0)
Ky = DirectionalGradient_Fixed((m,n),3,-0.61,dir=1)
u = int(np.sqrt(Kx.shape[0]))



for i,result_dir in enumerate(args.results_dirs):
    result_dir = Path(result_dir)
    if result_dir.exists() == False:
        raise Exception(f'{result_dir} doesnt exists')
    
    # Load results
    result_data = np.load(result_dir,allow_pickle=True)
    sol = result_data.item()['sol'].reshape(true_img.shape)
    param = result_data.item()['param']
    recons = exposure.match_histograms(sol,true_img)
    
    patch_size = len(param)
    
    ax[0,i+2].imshow(recons,cmap='gray')
    ax[0,i+2].set_title(f'{patch_size}')
    ax[0,i+2].set_xticklabels([])
    ax[0,i+2].set_xticks([])
    ax[0,i+2].set_yticks([])
    ax[0,i+2].set_yticklabels([])
    ax[0,i+2].set_xlabel(f'PSNR={psnr(true_img,recons):.4f}\nSSIM={ssim(true_img,recons,data_range=true_img.max() - true_img.min()):.4f}')
    
    if len(param) > 1:
        Q = CircularPatchOperator((u,u),patch_size)
        p = Q*param
        v = int(np.sqrt(len(p)))
        par = ax[1,i+2].imshow(p.reshape((v,v)),cmap='viridis')
        ax[1,i+2].set_xticklabels([])
        ax[1,i+2].set_xticks([])
        ax[1,i+2].set_yticks([])
        ax[1,i+2].set_yticklabels([])
        cb = plt.colorbar(par,ax=ax[1,i+2],orientation='horizontal')
    else:
        ax[1,i+2].set_xticklabels([])
        ax[1,i+2].set_xticks([])
        ax[1,i+2].set_yticks([])
        ax[1,i+2].set_yticklabels([])
        ax[1,i+2].axis('off')

if save == False:
    plt.show()
else:
    print(f'Saving plot to {args.output_dir}')
    output_dir = Path(args.output_dir)
    if output_dir.exists() == False:
        os.makedirs(output_dir)
    tikzplotlib.save(output_dir /'patch_plot.tex')
    