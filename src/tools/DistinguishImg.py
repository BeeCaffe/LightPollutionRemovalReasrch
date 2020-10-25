import cv2 as cv
import numpy as np
import torch
import os


def distinguish_image(path, high_path, low_path, threshold):
    """
    distinguish image by mean pixel value
    :param path: initial images path,type=string
    :param high_path: the path you store the high illumination image,type=string
    :param low_path: the path you store the low illumination image,type=string
    :param threshold: the threshold between low&high illumination image,type=int|float
    :return: None
    """
    # path = "C:/Users/comin/Desktop/highlightdata256/"
    imgs = os.listdir(path)  # images name list under path
    num_low, num_high = 0, 0
    for i in imgs:
        img = cv.imread(path + str(i))
        tensor_cv = torch.from_numpy(img)
        if tensor_cv.float().mean() >= threshold:
            cv.imwrite(high_path + str(i), img)
            num_high += 1
        else:
            cv.imwrite(low_path + str(i), img)
            num_low += 1

    print('low illumination numbers:%d,high illumination numbers:%d' % (num_low, num_high))


# example
init_path = "C:/Users/comin/Desktop/highlightdata256/"
t_high_path = "C:/Users/comin/Desktop/high/"
t_low_paht = "C:/Users/comin/Desktop/low/"
t_threshold = 40
distinguish_image(init_path, t_high_path, t_low_paht, t_threshold)
