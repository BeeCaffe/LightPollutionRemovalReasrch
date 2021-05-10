import os
import cv2
import time
from src.trad_methods import bimber
from src.trad_methods import tps
from src.ecn import CompensateImages
import cupy as cp
import numpy as np

speed_dir = r'I:\access\speed_test_imgs/'
speed_dirs = [
    speed_dir+'img_0128/',
    speed_dir + 'img_0256/',
    speed_dir + 'img_0512/',
    speed_dir + 'img_1024/',
    speed_dir + 'img_2048/',
]

file = open('./ours_speed_test.txt', 'w+')
## test different size
for img_difsz_dir in speed_dirs:
    sz = int(img_difsz_dir[-5:-1])
    names = os.listdir(img_difsz_dir)
    imgs = []
    for name in names:
        path = img_difsz_dir + name
        img = cv2.imread(path)
        imgs.append(img)
    times = []
    ## test ours speed
    st = time.time()
    CompensateImages.compenImage(
        input_images=img_difsz_dir,
        pth_path=r'F:\yandl\ECN\checkpoint\weights\CompeNet_5000_3000_2_0.001_0.2_800_l1+l2+ssim+vggLoss.pth',
        rows=sz,
        cols=sz
    )
    et = time.time()
    times.append(et-st)
    print('size: ({},{}), ours time: {}'.format(sz, sz, et-st))
    file.write('size:({},{}), ours time: {}\n'.format(sz, sz, times[0]))