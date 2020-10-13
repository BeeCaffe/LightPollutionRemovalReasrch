import cv2 as cv
import src.tools.Utils as util
import os
import math
import time

args = {
    "highLightPath": "res/high/",
    "lowLightPath": "res/low/",
    "trainPath": "datasets/firstdataset/train/",
    "testPath": "datasets/firstdataset/test/",
    "size": 64,
    "trainNumber": None,
    "testNumber": None
}


high_light_names = os.listdir(args["highLightPath"])
low_light_name = os.listdir((args["lowLightPath"]))
train_a_path = args["trainPath"]+"A/"
train_b_path = args["trainPath"]+"B/"
test_a_path = args["testPath"]+"A/"
test_b_path = args["testPath"]+"B/"
if not os.path.exists(train_a_path): os.makedirs(train_a_path)
if not os.path.exists(train_b_path): os.makedirs(train_b_path)
if not os.path.exists(test_a_path): os.makedirs(test_a_path)
if not os.path.exists(test_b_path): os.makedirs(test_b_path)

highlight_img_lists = []
lowlight_img_lists = []
for high_img_name in high_light_names:
    path = args["highLightPath"]+high_img_name
    img = cv.imread(path)
    imgs = util.SplitImage(img, args["size"])
    highlight_img_lists = highlight_img_lists + imgs

for low_img_name in low_light_name:
    path = args["lowLightPath"]+low_img_name
    img = cv.imread(path)
    imgs = util.SplitImage(img, args["size"])
    lowlight_img_lists = lowlight_img_lists+imgs

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











