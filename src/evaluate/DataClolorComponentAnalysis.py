import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import src.tools.Utils as util
import os
import cv2
import time
args=dict(
    imgPath=r'F:\yandl\CompenNet-plusplus3.0\data\dataset_512_no_gama\prj\train/',
    # title='Color Distribution Histogram Of Training Data'
    title=r'Image Histogram Of Projector Input',
    pixel_exband_ratio=3.75
)

def statisticsHisto(imgPath, title=args['title']):
    plt.xlabel('pixel value')
    plt.ylabel('number')
    # 添加标题
    plt.title(title)
    nameList = os.listdir(imgPath)
    imgs = []
    i = 0
    total = len(nameList)
    st = time.time()
    for name in nameList:
        img = cv2.imread(imgPath+name)
        imgs.append(img)
        util.process('reading image...', i, total, st, time.time())
        i += 1

    print('computing b channel...')
    hist_b = args['pixel_exband_ratio']*cv2.calcHist(imgs, [0], None, [256], [0, 255])
    print('computing g channel...')
    hist_g = args['pixel_exband_ratio']*cv2.calcHist(imgs, [1], None, [256], [0, 255])
    print('computing r channel...')
    hist_r = args['pixel_exband_ratio']*cv2.calcHist(imgs, [2], None, [256], [0, 255])
    print('plot b channel...')
    plt.plot(hist_b, label='B')
    print('plot g channel...')
    plt.plot(hist_g, label='G')
    print('plot t channel...')
    plt.plot(hist_r, label='R')
    plt.legend()
    plt.savefig('./temp.png')

if __name__=='__main__':
    statisticsHisto(args['imgPath'])