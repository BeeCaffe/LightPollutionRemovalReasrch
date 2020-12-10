import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train', train_number = None, gamma = None, img_size=None):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.num = train_number
        self.img_size = img_size
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        self.files_A = self.files_A[:self.num]
        self.files_B = self.files_B[:self.num]
        self.gamma = gamma

    def __getitem__(self, index):
        img = Image.open(self.files_A[index % len(self.files_A)])
        if self.img_size is not None:
            img = img.resize(self.img_size)
        if self.gamma is not None:
            img = self.GammaCorrection(img, self.gamma)
        item_A = self.transform(img)

        if self.unaligned:
            img = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
            if self.img_size is not None:
                img = img.resize(self.img_size)
            item_B = self.transform(img)
        else:
            img = Image.open(self.files_B[index % len(self.files_B)])
            if self.img_size is not None:
                img = img.resize(self.img_size)
            item_B = self.transform(img)
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def PIL2Numpy(self, img, dtype=np.float):
        return np.array(img, dtype=dtype)

    def Numpy2PIL(self, img):
        img = np.array(img, dtype=np.uint8)
        img = Image.fromarray(img)
        return img

    def GammaCorrection(self, img, gamma=2.2):
        img = self.PIL2Numpy(img, dtype=np.float)
        img = np.power(img, 1/gamma)
        img = self.Numpy2PIL(img)
        return img
