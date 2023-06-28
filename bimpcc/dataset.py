import numpy as np
import pylops
from PIL import Image
from bimpcc.operators import PatchSelection
from skimage.transform import resize

class NoiseDataset:
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
    
class InpaintingDataset:
    def __init__(self, dataset_dir, scale) -> None:
        self.dataset_dir = dataset_dir
        self.scale = scale
    def get_data(self,var=0.04,lost_percentage=0.3):
        print(f'Loading data from {self.dataset_dir} using a scale of {self.scale}...')
        if self.dataset_dir.exists() == False:
            raise Exception('Dataset directory does not exist.')
        true_imgs_path = [p for p in self.dataset_dir.glob('true*.png')]
        true_imgs = []
        noisy_imgs = []
        for image_path in true_imgs_path:
            np.random.seed(1234)
            pil_img = Image.open(image_path)
            pil_img = pil_img.resize((int(pil_img.width*self.scale),int(pil_img.height*self.scale)))
            img = np.array(pil_img.convert('L'))
            img = img / np.amax(img)
            noise = np.random.normal(loc=0,scale=var,size=img.shape)
            noisy = img + noise
            PSel = PatchSelection(img.shape, (int(lost_percentage*img.shape[0]),int(lost_percentage*img.shape[1])))
            dmg = PSel.matvec(noisy.ravel())
            # noisy = PSel.T.matvec(dmg).reshape(img.shape)
            true_imgs.append(img)
            noisy_imgs.append(dmg)
        return true_imgs[0],noisy_imgs[0],PSel
    
def load_shepp_logan_phantom(scale=1.0,subsampling=0.7):
    img = np.load('datasets/sheep_logan/phantom.npy')
    img = img/img.max()
    ny,nx = img.shape
    
    # Resize image
    # img_pil = Image.fromarray(img.astype('uint8'),'L')
    # print(img_pil)
    scaled_size = (int(nx * scale), int(ny * scale))
    # scaled_img = np.array(img_pil.resize(scaled_size))
    # print(scaled_img.shape, type(scaled_img))
    scaled_img = resize(img, scaled_size, anti_aliasing=True)
    
    # Sampling Operator
    np.random.seed(10)
    nxsub = int(np.round(np.prod(scaled_size)*subsampling))
    iava = np.sort(np.random.permutation(np.arange(np.prod(scaled_size)))[:nxsub])
    Rop = pylops.Restriction(np.prod(scaled_size),iava,axis=0,dtype=np.complex128)
    # Rop = pylops.Restriction(np.prod(scaled_size),iava,axis=0)
    print(f'{Rop=}')
    
    # 2D Fourier Operator
    Fop = pylops.signalprocessing.FFT2D(dims=scaled_size)
    print(f'{Fop=}')
    
    # Get synthetic measurements
    y = Rop * Fop * scaled_img.ravel()
    # print(f'{y=}, {y.shape=}')
    
    return scaled_img, y, Rop, Fop
            
        