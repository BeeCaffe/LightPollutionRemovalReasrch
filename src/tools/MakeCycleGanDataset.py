import cv2 as cv
import src.tools.Utils as util
import os
import math
import time
import random
import numpy as np

args = {
    "camPath": r"F:\yandl\LightPollutionRemovalReasrch\datasets\pairnet_geocored\cam\Warpping-Net_l1+l2+ssim_4000_256_50_0.001_0.2_5000_0.0001/",
    "prjPath": r"F:\yandl\LightPollutionRemovalReasrch\datasets\dataset_512_no_gama_new\prj\train/",
    "trainPath": "datasets/cyclecn-curve/train/",
    "testPath": "datasets/cyclecn-curve/test/",
    "size": 256,
    "trainNumber": 20000,
    "testNumber": 1000,
    "shuffle": True
}


high_light_names = os.listdir(args["camPath"])
low_light_name = os.listdir((args["prjPath"]))
train_a_path = args["trainPath"]+"A/"
train_b_path = args["trainPath"]+"B/"
test_a_path = args["testPath"]+"A/"
test_b_path = args["testPath"]+"B/"
if not os.path.exists(train_a_path): os.makedirs(train_a_path)
if not os.path.exists(train_b_path): os.makedirs(train_b_path)
if not os.path.exists(test_a_path): os.makedirs(test_a_path)
if not os.path.exists(test_b_path): os.makedirs(test_b_path)

def check_black_white(img):
    H, W, C = img.shape
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, 80)[1]
    pixels_white = cv.countNonZero(thresh)
    pixels_black = H*W - pixels_white
    if pixels_white < int(0.8*H*W) or pixels_black < int(0.8*H*W):
        return True
    return False

def split_image(img, size=256, gap=256):
    img_list = []
    height = img.shape[0]
    width = img.shape[1]
    h = int(height / size)
    w = int(width / size)
    for i in range(h):
        for j in range(w):
            for k in range(width//gap):
                tmp_img = img[i*size:(i+1)*size, j*size+k*gap:(j+1)*size+k*gap]
                width, height, channel = tmp_img.shape
                if width == height:
                    img_list.append(tmp_img)
    return img_list

highlight_img_lists = []
lowlight_img_lists = []
print("split high light image...")
for high_img_name in high_light_names:
    path = args["camPath"]+high_img_name
    img = cv.imread(path)
    # img = cv.resize(img, (512, 512))
    imgs = split_image(img, args["size"])
    highlight_img_lists = highlight_img_lists + imgs

print("split low light image...")
for low_img_name in low_light_name:
    path = args["prjPath"]+low_img_name
    img = cv.imread(path)
    imgs = split_image(img, args["size"])
    lowlight_img_lists = lowlight_img_lists+imgs

if args["shuffle"]:
    random.shuffle(highlight_img_lists)
    random.shuffle(lowlight_img_lists)

diff = int(math.fabs(len(highlight_img_lists)-len(lowlight_img_lists)))
if len(highlight_img_lists) > len(lowlight_img_lists):
    for i in range(0, diff):
        lowlight_img_lists.append(lowlight_img_lists[i])
elif len(highlight_img_lists) < len(lowlight_img_lists):
    for i in range(0, diff):
        highlight_img_lists.append(highlight_img_lists[i])

if args["trainNumber"] is None or args["testNumber"] is None:
    args["trainNumber"] = int(0.8*len(highlight_img_lists))
    args["testNumber"] = int(0.2*len(highlight_img_lists))

i = 0
total = len(highlight_img_lists)
st = time.time()
for img_high, img_low in zip(highlight_img_lists, lowlight_img_lists):
    if i<args["trainNumber"]:
        train_a = train_a_path+str(i)+".jpg"
        train_b = train_b_path+str(i)+".jpg"
        cv.imwrite(train_a, img_high)
        cv.imwrite(train_b, img_low)
    else:
        test_a = test_a_path+str(i)+".jpg"
        test_b = test_b_path+str(i)+".jpg"
        cv.imwrite(test_a, img_high)
        cv.imwrite(test_b, img_low)
    i+=1
    et = time.time()
    util.process("make data set: ", i, total, st, et)
print("Done!")











