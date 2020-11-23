import cv2
import os
import src.tools.Utils as utils
import numpy as np
import matplotlib.pyplot as plt
from pyheatmap.heatmap import HeatMap
from PIL import Image
args = dict(
    input_path="C:/canary/data/PaperData/input/",
    uncorrected_path="C:/canary/data/PaperData/hemispherical_corrected/uncorrected/",
    single_net_path = "C:/canary/data/PaperData/hemispherical_corrected/singlenet/gamma/gamma=1.4/",
    pair_net_path = "C:/canary/data/PaperData/hemispherical_corrected/pair-net/train_mask/1.4/",
    im_list=[4, 32, 9, 28],
)


def max_blur(img, ksize):
    width, height, channel = img.shape
    for i in range(0, width//ksize):
        for j in range(0, height//ksize):
            img[i*ksize:(i+1)*ksize, j*ksize:(j+1)*ksize] = np.mean(img[i*ksize:(i+1)*ksize, j*ksize:(j+1)*ksize])
    return img

def LightRemoveHeatMap(uncorrected, corrected):
    sub_img = np.subtract(uncorrected, corrected)
    # sub_img = max_blur(sub_img, 7)
    sub_img = np.sum(sub_img, axis=2, dtype=np.uint8)
    sub_img = np.interp(sub_img, (sub_img.min(), sub_img.max()), (0, 240))
    sub_img = np.array(sub_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(sub_img, cv2.COLORMAP_JET)
    return heat_img

def OverLayHeatMap(image, heatmap):
    overlay = image.copy()
    alpha = 0.6 # 设置覆盖图片的透明度
    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (255, 0, 0), -1)  # 设置蓝色为热度图基本色
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)  # 将背景热度图覆盖到原图
    image = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)  # 将热度图覆盖到原图
    return image

img2d=[]
for i in args['im_list']:
    imgs = []
    input = cv2.imread(args['input_path']+str(i)+".jpg")
    uncorrected = cv2.imread(args['uncorrected_path']+str(i)+".jpg")
    pairnet = cv2.imread(args['pair_net_path']+str(i)+".jpg")
    singlenet = cv2.imread(args['single_net_path']+str(i)+".jpg")

    pairnet_heatmap = LightRemoveHeatMap(uncorrected, pairnet)
    pairnet = OverLayHeatMap(input, pairnet_heatmap)
    singlenet_heatmap = LightRemoveHeatMap(uncorrected, singlenet)
    singlenet = OverLayHeatMap(input, singlenet_heatmap)

    imgs.append(utils.ImageAddTag( cv2.imread(args['input_path']+str(i)+".jpg"), "a"))
    imgs.append(utils.ImageAddTag(pairnet, "b"))
    imgs.append(utils.ImageAddTag(singlenet, "c"))
    img2d.append(imgs)
img = utils.CombineImages2D(img2d)
cv2.imwrite("./attention_heatmap.jpg", img)
