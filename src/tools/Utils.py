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
            img_list.append(img[i * size:(i + 1) * size, j * size:(j + 1) * size])
    return img_list
