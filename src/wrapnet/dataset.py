import os
from torch.utils.data import Dataset
import cv2 as cv
import utils
class SAUnet_Dataset(Dataset):
    def __init__(self, data_root=None, index=None, size=None):
        self.data_root = data_root
        self.size = size
        img_list = sorted(os.listdir(self.data_root))
        if index is not None: img_list = [img_list[x] for x in index]
        self.img_names = [utils.join(self.data_root, name) for name in img_list]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        assert os.path.isfile(img_name), img_name + ' does not exist'
        img = cv.imread(self.img_names[idx])
        # resize image if size is specified
        if self.size is not None:
            img = cv.resize(img, self.size[::-1])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img
