import cv2
import numpy as np
import os
import src.wrapnet.GetWarppedImage as geo
import src.tools.Utils as utils
import src.tools.GetRegion as GetRegion
debug = False
args = dict(
    projected_img_path=r'I:\GraduationThesis\SubstractNet\diff_resblocks\projectted\1/',
    save_dir=r'I:\GraduationThesis\SubstractNet\diff_resblocks\geocorrectted\1/',
    weight_path='checkpoint/Warpping-Net-1920-1080_l1+l2+ssim_50_4_200.pth',
    cor_pt=(1400, 950),
    cor_size=(2000, 3000)
)

def ReadImage():
    utils.rename(args['projected_img_path'])
    prj_name_list = os.listdir(args['projected_img_path'])
    prj_name_list.sort(key=lambda x: int(x[:-4]))
    sorted(prj_name_list)
    prj_imgs = []
    for prj_name in prj_name_list:
        path = args['projected_img_path'] + prj_name
        img = cv2.imread(path)
        prj_imgs.append(img)
    return prj_imgs

def CorrectImage(prj_imgs):
    for i in range(0, len(prj_imgs)):
        prj_imgs[i] = GetRegion.GetRegion2(prj_imgs[i],
                                             pt=args['cor_pt'],
                                             rows=args['cor_size'][0],
                                             cols=args['cor_size'][1])
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