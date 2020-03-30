from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy
import pandas
import skimage.transform
from matplotlib import pyplot as plt
from bananas.dataset import Feature, DataSet
from bananas.sampling import RandomSampler
from bananas.utils import images

import imgaug
from imgaug import augmenters
from torchvision import transforms
from datamodels import Template


# Valid extensions for image files
IMG_EXTENSIONS = ('.jpg', '.png')


def imshow(img: numpy.ndarray):
    ''' Utility function used to display image ''' 
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')

    
def tpl_load(tpl_path: Path) -> Template:
    ''' Load image template given its path '''
    name = tpl_path.name.split('.')[0]
    tpl = images.open_image(str(tpl_path), convert='L')
    tpl_width, tpl_height = tpl.shape[::-1]
    return Template(
        name=name,
        path=tpl_path,
        image=tpl,
        width=tpl_width,
        height=tpl_height)


def tpl_load_all(tpl_folder: Path) -> List[Template]:
    ''' Load all templates given a folder containing their images '''
    return [tpl_load(f) for f in tpl_folder.iterdir() if f.suffix in IMG_EXTENSIONS]


def split_datasets(df: pandas.DataFrame, test_size: int,
                   target_col: str = 'diagnosis', random_seed: int = 0) -> \
                   Tuple[pandas.DataFrame, pandas.DataFrame]:
    ''' Splits the dataset into train and test randomly '''

    # Identify left-out subset of keys
    all_keys = df.index.to_list()
    sampler = RandomSampler(all_keys, random_seed=random_seed)
    test_keys = sampler(batch_size=test_size)
    train_keys = [key for key in all_keys if key not in test_keys]

    # Create and return the corresponding dataframes
    return df.loc[train_keys], df.loc[test_keys]


class ImageLoader:
    ''' Load images from a list of paths '''
    
    def __init__(
        self,
        arr: List[str],
        cache: int = 0,
        resize: Tuple[int] = None,
        normalize: bool = False,
        **img_opts):
        self.arr = arr
        self._cache = OrderedDict()
        self._cache_size = cache
        self._cache_flag = cache > 0
        self._resize = resize
        self._normalize = normalize
        self._img_opts = img_opts

    def load(self, impath: str, process: bool = True):

        # Load & save image from cache if option is enabled
        img = self._cache.get(
            impath, images.open_image(impath, channels=True, **self._img_opts))
        if self._cache_flag and impath not in self._cache:
          self._cache[impath] = img
          if len(self._cache) > self._cache_size:
              self._cache.popitem(last=False)
            
        # Resize image if requested
        if self._resize is not None:
            img = skimage.transform.resize(
                img.astype(float),
                self._resize,
                mode='constant').astype(img.dtype)
            
        # Normalize image if requested
        if process and self._normalize and not self._img_opts.get('uint8'):
            img = images.normalize(
                img, means=[0.485, 0.456, 0.406], stdevs=[0.229, 0.224, 0.225])

        return img

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        return self.load(self.arr[idx])


class ImageAugmenterLoader(ImageLoader):
    ''' Load an image given its path and augment it using random transformations '''
    
    def __init__(
        self,
        arr: List,
        cache: int = 0,
        resize: Tuple[int] = None,
        normalize: bool = False,
        **img_opts):
        super().__init__(
          arr, cache=cache, resize=resize, normalize=normalize, uint8=True, **img_opts)
        self._augmenter = augmenters.Sequential([
            augmenters.Crop(percent=(0, .1)),
            augmenters.Sometimes(.5, augmenters.GaussianBlur(sigma = (0, .5))),
            augmenters.Multiply((.9, 1.1), per_channel = 0.2),
            augmenters.Affine(
                scale={"x": (.9, 1.1), "y": (.9, 1.1)},
                translate_percent={"x": (-.1, .1), "y": (-.1, .1)},
                rotate=(-10, 10),
                shear=(-5, 5)
            ),
        ], random_order = True)
        
    def _augment(self, img):
        assert img.ndim == 3, 'Color channel required. Dimensions: %d' % img.ndim
        img = numpy.moveaxis(img, 0, -1)
        img = self._augmenter.augment_image(img)
        img = numpy.moveaxis(img, -1, 0)
        img = img.astype(float) / 255
        if self._normalize:
            img = images.normalize(
                img, means=[0.485, 0.456, 0.406], stdevs=[0.229, 0.224, 0.225])

        return img

    def __getitem__(self, idx):
        img = self.load(self.arr[idx], process=False)
        return self._augment(img)


class ImageMultiLoader(ImageLoader):
    ''' Loads images from input array containing a list of images for each datapoint '''
    
    def __getitem__(self, idx):
        imgs = [self.load(impath) for impath in self.arr[idx]]
        imgs = [img.reshape(*img.shape[1:]) for img in imgs]
        return numpy.array(imgs)


class ImageAugmenterMultiLoader(ImageAugmenterLoader):
    ''' Same as ImageMultiLoader but with image augmentation '''

    def __getitem__(self, idx):
        # Load images from disk
        imgs = [self.load(impath) for impath in self.arr[idx]]
        # Augment each image independently
        imgs = [self._augment(img) for img in imgs]
        # Remove the grayscale color channel
        imgs = [img.reshape(*img.shape[1:]) if img.shape[0] == 1 else img
                for img in imgs]
        # Join all images into a single array
        return numpy.array(imgs)


class NpyLoader:
    ''' Load npy files from a list of paths '''
    def __init__(self, arr: List[str]):
        self.arr = arr

    def __len__(self):
        return len(self.arr)

    def load(self, npypath: str):
        # https://github.com/numpy/numpy/issues/12847
        npy = numpy.load(npypath, mmap_mode='r')
        data = npy[:]
        del npy
        return data

    def __getitem__(self, idx):
        return self.load(self.arr[idx])


class NpyMultiLoader(NpyLoader):
    ''' Loads npy files from input array containing a list of paths for each datapoint '''

    def __getitem__(self, idx):
        npys = [self.load(npypath) for npypath in self.arr[idx]]
        #npys = [npy.reshape(*npy.shape[1:]) for npy in npys]
        return numpy.array(npys)
