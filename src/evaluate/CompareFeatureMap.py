import cv2
import os
import src.tools.Utils as utils
args = dict(
    input_path="C:/canary/data/PaperData/input/",
    white_mask_path="C:/canary/data/PaperData/hemispherical_corrected/pair-net/feature-map/1-2mask/",
    black_mask_path="C:/canary/data/PaperData/hemispherical_corrected/pair-net/feature-map/2-mask/",
    im_list=[4, 32, 9, 28],
)
imgs_2d = []
for i in args['im_list']:
    input = args['input_path']+str(i)+".jpg"
    imgs = []
    imgs.append(utils.ImageAddTag(cv2.imread(input), 'a'))
    imgs.append(utils.ImageAddTag(cv2.imread(args['black_mask_path']+str(i)+".png"), 'b'))
    imgs.append(utils.ImageAddTag(cv2.imread(args['white_mask_path']+str(i)+".png"), "c"))
    heat_maps = []
    imgs_2d.append(imgs)
img = utils.CombineImages2D(imgs_2d)
cv2.imwrite("./feature_map.jpg", img)

