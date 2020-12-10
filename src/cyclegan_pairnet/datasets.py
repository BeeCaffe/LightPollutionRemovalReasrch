import glob
import random
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
class ImageDataset(Dataset):
    def __init__(self, root,  unaligned=False, mode='train', train_number=None, img_size=None):
        self.unaligned = unaligned
        self.num = train_number
        self.img_size = img_size
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        self.files_A = self.files_A[:self.num]
        self.files_B = self.files_B[:self.num]

    def __getitem__(self, index):
        img = cv2.imread(self.files_A[index % len(self.files_A)])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.img_size is not None:
            img = cv2.resize(img, self.img_size)
        item_A = img

        if self.unaligned:
            img = cv2.imread(self.files_A[index % len(self.files_B)])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.img_size is not None:
                img = img.resize(self.img_size)
            item_B = img
        else:
            img = cv2.imread(self.files_A[index % len(self.files_B)])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.img_size is not None:
                img = img.resize(self.img_size)
            item_B = img
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def normalize(self):
        pass

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
