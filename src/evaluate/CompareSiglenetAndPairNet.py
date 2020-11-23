import cv2
import os
import src.tools.Utils as utils
args = dict(
    input_path="C:/canary/data/PaperData/input/",
    uncorrected_path="C:/canary/data/PaperData/hemispherical_corrected/uncorrected/",
    single_path="C:/canary/data/PaperData/hemispherical_corrected/singlenet/gamma/gamma=1.4/",
    pairnet_path="C:/canary/data/PaperData/hemispherical_corrected/pair-net/train_mask/1.4/",
    im_list=[4, 32, 9, 28],
)
imgs_2d = []
for i in args['im_list']:
    input = args['input_path']+str(i)+".jpg"
    imgs = []
    imgs.append(utils.ImageAddTag(cv2.imread(input), 'a'))
    imgs.append(utils.ImageAddTag(cv2.imread(args['uncorrected_path']+str(i)+".jpg"), 'b'))
    imgs.append(utils.ImageAddTag(cv2.imread(args['pairnet_path']+str(i)+".jpg"), "c"))
    imgs.append(utils.ImageAddTag(cv2.imread(args['single_path']+str(i)+".jpg"), "d"))
    heat_maps = []
    for img in imgs:
        heat_maps.append(utils.OriginHeatMap(img))
    imgs_2d.append(imgs)
    imgs_2d.append(heat_maps)
img = utils.CombineImages2D(imgs_2d)
cv2.imwrite("./compare_single_pair.jpg", img)

