import cv2
import numpy as np
import os
import src.wrapnet as warpnet
import src.tools.Utils as utils
import src.tools.GetRegion as GetRegion
args = dict(
    input_img_path='',
    projected_img_path='',
    width=1920,
    height=1080,
    need_geo_cor=True,
    cor_pt=(570, 210),
    cor_size=(820, 450)
)

def ReadImage():
    input_name_list = os.listdir(args['input_img_path'])
    prj_name_list = os.listdir(args['projected_img_path'])
    input_imgs = []
    prj_imgs = []
    for in_name in input_name_list:
        path = args['input_img_path'] + in_name
        img = cv2.resize(cv2.imread(path), (args['width'], args['height']))
        input_imgs.append(img)
    for prj_name in prj_name_list:
        path = args['projected_img_path'] + prj_name
        img = cv2.resize(cv2.imread(path), (args['width'], args['height']))
        prj_imgs.append(img)
    return input_imgs, prj_imgs

def CorrectImage(input_imgs, prj_imgs):
    if len(input_imgs) != len(prj_imgs):
        print("error!, input image number not equal to prjected image")
        exit(0)
    for i in range(0, len(input_imgs)):
        input_imgs[i] = GetRegion.GetRegion2(input_imgs[i],
                                             pt=args['cor_pt'],
                                             width=args['cor_size'][0],
                                             height=args['cor_size'][1])
        prj_imgs[i] = GetRegion.GetRegion2(prj_imgs[i],
                                             pt=args['cor_pt'],
                                             width=args['cor_size'][0],
                                             height=args['cor_size'][1])
    return input_imgs, prj_imgs

def CalculateObjectiveEvaluationIndex():
    input_imgs, prj_imgs = ReadImage()
    input_imgs, prj_imgs = CorrectImage(input_imgs, prj_imgs)
    psnr_lists = []
    ssim_lists = []
    rmse_lists = []
    for x, y in zip(input_imgs, prj_imgs):
        psnr = utils.psnr(x, y)
        ssim = utils.ssim(x, y)
        rmse = utils.rmse(x, y)
        psnr_lists.append(psnr)
        ssim_lists.append(ssim)
        rmse_lists.append(rmse)
    psnr = np.average(psnr_lists)
    ssim = np.average(ssim_lists)
    rmse = np.average(rmse_lists)
    print("psnr: {}, ssim: {}, rmse: {}".format(psnr, ssim, rmse))
    return psnr, ssim, rmse

if __name__=='__main__':
    CalculateObjectiveEvaluationIndex()