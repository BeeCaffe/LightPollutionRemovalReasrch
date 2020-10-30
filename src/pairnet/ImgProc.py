import numpy as np
import cv2 as cv
import torch
import src.pairnet.utils as utils
import src.pairnet.ImgProc as ImgProc
DEBUG=False
# find the projector FOV mask
def get_mask_corner(cam_ref_path,train_option):
    '''
    :brief : get the mask corner and projector fov mask by camera reference image with put in directory 'cam/ref'
    :param cam_ref_path: 'cam/ref'
    :param train_option: arguments of train
    :return: mask_corner,prj_fov_mask
    '''
    cam_surf = utils.readImgsMT(cam_ref_path, index=[0], size=train_option['train_size'])
    im_diff = utils.readImgsMT(cam_ref_path, index=[0], size=train_option['train_size'])

    if DEBUG:
        img = cam_surf[0].permute(1, 2, 0).mul(255).to('cpu').numpy()
        img = np.uint8(img)
        cv.imshow('cam_surf img'
                  '', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    if DEBUG:
        img = im_diff[0].permute(1, 2, 0).mul(255).to('cpu').numpy()
        img = np.uint8(img)
        cv.imshow('im_diff img'
                  '', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    im_diff = im_diff.numpy().transpose((2, 3, 1, 0))
    # print(im_diff.shape)
    prj_fov_mask = torch.zeros(cam_surf.shape)
    # threshold im_diff with Otsu's method
    mask_corners = [None] * im_diff.shape[-1]
    for i in range(im_diff.shape[-1]):
        im_mask, mask_corners[i] = ImgProc.thresh(im_diff[:, :, :, i])
        prj_fov_mask[i, :, :, :] = utils.repeat_np(torch.Tensor(np.uint8(im_mask)).unsqueeze(0), 3, 0)
    prj_fov_mask = prj_fov_mask.byte()

    if DEBUG:
        img = prj_fov_mask[0].permute(1, 2, 0).mul(255).to('cpu').numpy()
        img = np.uint8(img)
        cv.imshow('prj_fov_mask img'
                  '', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    print("mask corners:", mask_corners)
    return mask_corners, prj_fov_mask

def thresh(im_in):
    # threshold im_diff with Otsu's method
    if im_in.ndim == 3:
        im_in = cv.cvtColor(im_in, cv.COLOR_BGR2GRAY)
    if im_in.dtype == 'float32':
        im_in = np.uint8(im_in * 255)
    # _, im_mask = cv.threshold(cv.GaussianBlur(im_in, (5, 5), 0), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, im_mask = cv.threshold(im_in, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    im_mask = im_mask > 0
    # find the largest contour by area then convert it to convex hull
    contours, hierarchy = cv.findContours(np.uint8(im_mask), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hulls = cv.convexHull(max(contours, key=cv.contourArea))
    im_mask = cv.fillConvexPoly(np.zeros_like(im_mask, dtype=np.uint8), hulls, True) > 0

    # also calculate the bounding box
    bbox = cv.boundingRect(max(contours, key=cv.contourArea))
    corners = [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]], [bbox[0], bbox[1] + bbox[3]]]
    if DEBUG:
        for pt in corners:
            cv.circle(im_in,(pt[0],pt[1]),10,(0,0,255),-1)
    if DEBUG:
        print(corners)
        img = im_in
        img = np.uint8(img)
        cv.imshow('im_in img'
                   '', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    if DEBUG:
        img = im_mask
        img = np.uint8(img)
        cv.imshow('im_mask img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # normalize to (-1, 1) following pytorch grid_sample coordinate system
    h = im_in.shape[0]
    w = im_in.shape[1]

    for pt in corners:
        pt[0] = 2 * (pt[0] / w) - 1
        pt[1] = 2 * (pt[1] / h) - 1

    return im_mask, corners
