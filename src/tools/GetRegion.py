import os
import cv2
import src.tools.Utils as utils

def GetRegion(filepath, save_path, pt=(570, 210), width=820, height=450):
    name_lists = os.listdir(filepath)
    for name in name_lists:
        img_path = filepath+name
        img = cv2.resize(cv2.imread(img_path), (1920, 1080))
        img = utils.GetRegion(img, pt, width, height)
        cv2.imwrite(save_path+name, img)
        print("get region one image!")

def GetRegion2(img, pt=(570, 210), width=820, height=450):
    img = utils.GetRegion(img, pt, width, height)
    return img


if __name__=='__main__':
    GetRegion(filepath='C:/canary/data/PaperData/hemispherical/uncorrected/',
              save_path='C:/canary/data/PaperData/hemispherical_region/uncorrected/',
              pt=(570, 210),
              width=820,
              height=450)