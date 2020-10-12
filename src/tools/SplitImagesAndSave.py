import cv2 as cv
import os
import src.tools.Utils as util
'''
@:Function: split images to 256x256 images
'''
args = {
    "imgpath": "res/highlitdata",      #image path
    "savePath": "res/highlightdata256"      #save image path
}

name_lists = os.listdir(args["imgpath"])
imgs = []
for img_name in name_lists:
    img_path = args["imgpath"]+"/"+img_name
    img = cv.imread(img_path)
    img_lists = util.SplitImage(img, 256)
    imgs = imgs + img_lists
i = 1
for img in imgs:
    cv.imwrite(args["savePath"]+"/"+str(i)+".jpg", img)
    i += 1
print("Done!")