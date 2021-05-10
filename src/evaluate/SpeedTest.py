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
    speed_dir + 'img_0720/',
    speed_dir + 'img_1024/',
    speed_dir + 'img_2048/',
]

file = open('./speed_test.txt', 'w+')
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

    # ## test bimber speed
    # st = time.time()
    # bim = bimber.Bimber(imRoot=img_difsz_dir)
    # bim.compensateImgs()
    # et = time.time()
    # times.append(et-st)
    # print('size: ({},{}), bimber time: {}'.format(sz, sz, et-st))
    #
    # ## test tps speed
    # tps_obj = tps.TPS(input_img_dir=img_difsz_dir)
    # W = np.loadtxt(r'F:\tps_weight/w_0x31.txt')
    # st = time.time()
    # for img in imgs:
    #     tps_obj.TestSpeed(weight_mat = cp.array(W), img=cp.array(img))
    # et = time.time()
    # times.append(et-st)
    # print('size: ({},{}), tps time: {}'.format(sz, sz, et-st))
    print(sz)
    CompensateImages.compenImage(
        input_images=img_difsz_dir,
        pth_path=r'F:\yandl\ECN\checkpoint\weights\CompeNet_5000_3000_2_0.001_0.2_800_l1+l2+ssim+vggLoss.pth',
        rows=sz,
        cols=sz
    )

    ## test ours speed
    t = CompensateImages.compenImage(
        input_images=img_difsz_dir,
        pth_path=r'F:\yandl\ECN\checkpoint\weights\CompeNet_5000_3000_2_0.001_0.2_800_l1+l2+ssim+vggLoss.pth',
        rows=sz,
        cols=sz
    )
    our_fps = 1/(t/51)
    print('size: ({},{}), ours fps: {}'.format(sz, sz,
                                               our_fps))

    ## test compennet speed
    t = CompensateImages.compenImage(
        input_images=img_difsz_dir,
        pth_path=r'F:\yandl\ECN\checkpoint\weights\CompeNet_5000_3000_2_0.001_0.2_800_l1+l2+ssim.pth',
        rows=sz,
        cols=sz
    )
    compennet_fps = 1/(t/51)
    print('size: ({},{}), compennet fps: {}'.format(sz, sz,compennet_fps))
    file.write('size:({},{}), ours fps: {}, compennet fps: {}\n'.format(sz, sz, our_fps, compennet_fps))