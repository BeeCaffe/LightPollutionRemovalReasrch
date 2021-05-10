import os
import cv2
import src.tools.Utils as utils

def GetRegion(filepath, save_path, pt=(818, 1922), rows=820, cols=450):
    name_lists = os.listdir(filepath)
    for name in name_lists:
        img_path = filepath+name
        img = cv2.imread(img_path)
        img = utils.GetRegion(img, pt, rows, cols)
        cv2.imwrite(save_path+name, img)
        print("get region one image!")

def GetRegion2(img, pt=(570, 210), rows=820, cols=450):
    img = utils.GetRegion(img, pt, rows, cols)
    return img

if __name__=='__main__':
    GetRegion(filepath=r'I:\GraduationThesis\SubstractNet\diff_resblocks\projectted\1/',
              save_path=r'F:\yandl\LightPollutionRemovalReasrch\datasets\geocor\cam\train/',
              pt=(1400, 950),
              rows=2000,
              cols=3000)