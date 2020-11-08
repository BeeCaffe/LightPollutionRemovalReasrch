import cv2
import src.tools.Utils as utils
import os
args = dict(
    img_path="C:/canary/data/PaperData/single-net/residel-comparison/",
    combine_size=[2, 2],
    tags=["a", "b", "c", "d"],
    size=(1920, 1080)
)

img_names = os.listdir(args['img_path'])
if len(img_names) != args['combine_size'][0]*args['combine_size'][1]:
    print("image number error!")
    exit(0)

imgs = []
for name in img_names:
    path = args['img_path']+name
    img = cv2.resize(cv2.imread(path), args['size'])
    imgs.append(img)
real_imgs = []
for i in range(0, args['combine_size'][0]):
    temp_imgs = []
    for j in range(0, args['combine_size'][1]):
        idx = i*args['combine_size'][0]+j
        img = imgs[idx]
        img = utils.ImageAddTag(img, args['tags'][idx])
        temp_imgs.append(img)
    real_imgs.append(temp_imgs)

img = utils.CombineImages2D(real_imgs)
cv2.imwrite("./img.jpg", img)