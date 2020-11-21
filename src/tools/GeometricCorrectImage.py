import cv2
import numpy as np
import os
import src.wrapnet.GetWarppedImage as geo
import src.tools.Utils as utils
import src.tools.GetRegion as GetRegion
debug = False
args = dict(
    projected_img_path='I:\hemispherical\pair-net\\untrain_mask\\1.4\\',
    save_dir='I:\hemispherical_corrected\pair-net\\untrain_mask\\1.4\\',
    weight_path='checkpoint/Warpping-Net-1920-1080_l1+l2+ssim_50_4_100.pth',
    width=1920,
    height=1080,
    cor_pt=(570, 210),
    cor_size=(820, 450)
)

def ReadImage():
    prj_name_list = os.listdir(args['projected_img_path'])
    prj_name_list.sort(key=lambda x: int(x[:-4]))
    sorted(prj_name_list)
    prj_imgs = []
    for prj_name in prj_name_list:
        path = args['projected_img_path'] + prj_name
        img = cv2.resize(cv2.imread(path), (args['width'], args['height']))
        prj_imgs.append(img)
    return prj_imgs

def CorrectImage(prj_imgs):
    for i in range(0, len(prj_imgs)):
        prj_imgs[i] = GetRegion.GetRegion2(prj_imgs[i],
                                             pt=args['cor_pt'],
                                             width=args['cor_size'][0],
                                             height=args['cor_size'][1])
    prj_imgs = geo.cmpImage(args['weight_path'], prj_imgs, size=(1920, 1080))
    return prj_imgs

def SaveImages(prj_imgs):
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
    i = 0
    for img in prj_imgs:
        path = args['save_dir']+str(i)+'.jpg'
        cv2.imwrite(path, img)
        i+=1
        print("write {} images ...".format(i))
    print("Done!")

def GeoCorrectImage():
    prj_imgs = ReadImage()
    prj_imgs = CorrectImage(prj_imgs)
    SaveImages(prj_imgs)

if __name__=='__main__':
    GeoCorrectImage()