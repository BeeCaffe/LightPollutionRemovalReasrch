import os
import torch
import cv2 as cv
import numpy as np
import src.ecn.ECN as ECN
import time
MaxNum = 2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
paths = {
    'surf_path': 'F:\yandl\CompenNet-plusplus3.0\data\dataset_512_gama2.2\cmp\surf\surf.jpg',#the surface image path
}

def compenImage(input_images = None, pth_path = None, rows= None, cols = None):
    #get the data root of GroundTruth's pictures
    surf_img = cv.cvtColor(cv.resize(cv.imread(paths['surf_path']), (rows, cols)), cv.COLOR_BGR2RGB)
    surf_img = torch.from_numpy(surf_img).expand([1, rows, cols, 3])
    surf_img = surf_img.permute((0, 3, 1, 2)).float().div(255)
    #read each pictures in prjector data root
    name_list = os.listdir(input_images)
    name_list.sort(key=lambda x: int(x[:-4]))
    model = ECN.ECN().cuda()
    model.load_state_dict(torch.load(pth_path))
    prj_imgs = []
    for i, name in enumerate(name_list):
        #read each picture and feed it to CompeNet Model
        prj_img = cv.cvtColor(cv.imread(input_images + name), cv.COLOR_BGR2RGB)
        prj_imgs.append(prj_img)
    st = time.time()

    for prj_img in prj_imgs:
        prj_img = cv.resize(prj_img, (rows, cols))
        prj_img = torch.from_numpy(prj_img).expand([1, rows, cols, 3])
        prj_img = prj_img.permute((0, 3, 1, 2)).float().div(255)
        #load model
        pred = model(prj_img.to(device), surf_img.to(device))
        #save pred_l1_125_5000 to you desired path
        pred = pred[0, :, :, :]
        pred = np.uint8((pred[:, :, :] * 255).permute(1, 2, 0).cpu().detach().numpy())
        pred = pred[:, :, ::-1]
    et = time.time()
    return et-st
