import cv2
import numpy as np
import os
import src.wrapnet.GetWarppedImage as geo
import src.tools.Utils as utils
import src.tools.GetRegion as GetRegion
args = dict(
    input_img_path='res/input/',
    projected_img_path='data/geocor/cmp/res/Warpping-Net-1920-1080_l1+l2+ssim_50_4_100/',
    weight_path='checkpoint/Warpping-Net-1920-1080_l1+l2+ssim_50_4_100.pth',
    width=1920,
    height=1080,
    need_geo_cor=False,
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
        print("error!, input image number: {} not equal to prjected image number: {}"
              .format(len(input_imgs), len(prj_imgs)))
        exit(0)
    for i in range(0, len(input_imgs)):
        input_img = GetRegion.GetRegion2(input_imgs[i],
                                             pt=args['cor_pt'],
                                             width=args['cor_size'][0],
                                             height=args['cor_size'][1])
        input_imgs[i] = geo.cmpImage(args['weight_path'], input_img)
        prj_img = GetRegion.GetRegion2(prj_imgs[i],
                                             pt=args['cor_pt'],
                                             width=args['cor_size'][0],
                                             height=args['cor_size'][1])
        prj_imgs[i] = geo.cmpImage(args['weight_path'], prj_img)
    return input_imgs, prj_imgs

def CalculateObjectiveEvaluationIndex():
    input_imgs, prj_imgs = ReadImage()
    if args['need_geo_cor']:
        input_imgs, prj_imgs = CorrectImage(input_imgs, prj_imgs)
    psnr_lists = []
    ssim_lists = []
    rmse_lists = []
    for x, y in zip(prj_imgs, input_imgs):
    #     cv2.imshow("img", x)
    #     cv2.waitKey(0)
    #     cv2.imshow("img", y)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
        psnr = utils.psnr(x, y)
        ssim = utils.ssim(x, y)
        rmse = utils.rmse(x, y)
        psnr_lists.append(psnr)
        ssim_lists.append(ssim)
        rmse_lists.append(rmse)
        print("psnr: {}, ssim: {}, rmse: {}".format(psnr, ssim, rmse))
    psnr = np.average(psnr_lists)
    ssim = np.average(ssim_lists)
    rmse = np.average(rmse_lists)
    print("psnr: {}, ssim: {}, rmse: {}".format(psnr, ssim, rmse))
    print("Done!")
    return psnr, ssim, rmse

if __name__=='__main__':
    CalculateObjectiveEvaluationIndex()