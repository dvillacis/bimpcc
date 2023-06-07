import numpy as np
from PIL import Image

class Dataset:
    def __init__(self, dataset_dir, scale) -> None:
        self.dataset_dir = dataset_dir
        self.scale = scale
    def get_data(self,var=0.05):
        print(f'Loading data from {self.dataset_dir} using a scale of {self.scale}...')
        if self.dataset_dir.exists() == False:
            raise Exception('Dataset directory does not exist.')
        true_imgs_path = [p for p in self.dataset_dir.glob('true*.png')]
        true_imgs = []
        noisy_imgs = []
        for image_path in true_imgs_path:
            pil_img = Image.open(image_path)
            pil_img = pil_img.resize((int(pil_img.width*self.scale),int(pil_img.height*self.scale)))
            img = np.array(pil_img.convert('L'))
            img = img / np.amax(img)
            noise = np.random.normal(loc=0,scale=var,size=img.shape)
            noisy = img + noise
            true_imgs.append(img)
            noisy_imgs.append(noisy)
        return true_imgs[0],noisy_imgs[0]
            
        