import cv2 as cv
import os
import numpy as np

'''
@:param img -> cv image
@:param size -> the image size of you suppose to split
@:return img_list -> splitted images list
'''


def SplitImage(img, size=256):
    img_list = []
    height = img.shape[0]
    width = img.shape[1]
    h = int(height / size)
    w = int(width / size)
    for i in range(h):
        for j in range(w):
            img_list.append(img[i*size:(i+1)*size, j*size:(j+1)*size])
    return img_list


def join(path1,path2):
    if not path1.endswith('/'):
        path1 += '/'
    return path1+path2

"""
@ brief : show the process status
:param title: name of this process follow you want.
:param now: the number of now
:param total: total number
:param startTm: start time
:param nowTm: now time
:return:
"""
def process(title,now,total,startTm,nowTm):
    rate = (now/total)*100
    h = int((-startTm+nowTm)//3600)
    m = int(((-startTm+nowTm)%3600)//60)
    s = int((-startTm+nowTm)%60)
    print('{} : rate {:<2.2f}% time {:<2d}:{:<2d}:{:<2d} ['.format(title, rate, h, m, s), end='')
    for i in range(0, int((now/total)*100)):
        print('#', end='')
    for i in range(int((now/total)*100), 100):
        print(' ', end='')
    print(']\r')
