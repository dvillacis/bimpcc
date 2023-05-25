# Solving Bilevel Parameter Learning Problems as MPCC
This code implements the experiments presented in the paper:
...

The main idea is to reformulate a nonsmooth bilevel parameter learning problem as a Mathematical Program with Complementarity Constraints

## Prerequisites
This python module requires the following modules:
* numpy
* pylops
* pyproximal
* ipopt with SPRAL solver
* pyoptsparse
* pillow
* scikit-image

## Installation
It is necessary to install the module using pip in developer mode. Once the repository is cloned, cd into the folder and execute
```bash
$ cd bimpcc
$ pip install -e .
```

## Run MPCC Bilevel Parameter Learning
### TV Denoising
```bash
$ python experiments/tv_denoising.py $dataset_folder $output_folder --tik $tikhonov_value --patch_size $patch_size --img_scale $img_scale
```
### TV Inpainting
```bash
$ python experiments/tv_inpainting.py $dataset_folder $output_folder --tik $tikhonov_value --patch_size $patch_size
```

### Directional TV (DTV) Denoising
```bash
$ python experiments/dtv_denoising.py $dataset_folder $output_folder --tik $tikhonov_value --patch_size $patch_size --angle $angle_diffusion
```

### Directional TV (DTV) Inpainting
```bash
$ python experiments/dtv_inpainting.py $dataset_folder $output_folder --tik $tikhonov_value --patch_size $patch_size --angle $angle_diffusion
```

### MRI TV Reconstruction
```bash
$ python experiments/mri_reconstruciton.py $dataset_folder $output_folder --tik $tikhonov_value --patch_size $patch_size --sampling_type $sampling_type --sampling_perc $sampling_perc
```
