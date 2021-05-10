import cv2
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

l1_fun = nn.L1Loss()
l2_fun = nn.MSELoss()

def torch2cv(tensor):
    return tensor.permute(0, 2, 3, 1).mul(255).to('cpu').detach().numpy()

def cv2torch(imgs):
    imgs = torch.from_numpy(imgs)
    return imgs.permute((0, 3, 1, 2)).float().div(255).to('cuda')

def hsvloss(predict_tensor, ground_truth_tensor):
    predict = torch2cv(predict_tensor)
    ground_truth = torch2cv(ground_truth_tensor)
    for i in range(predict.shape[0]):
        predict[i] = cv2.cvtColor(predict[i], cv2.COLOR_BGR2HSV)
        ground_truth[i] = cv2.cvtColor(ground_truth[i], cv2.COLOR_BGR2HSV)
    predict_tensor = cv2torch(predict)
    ground_truth_tensor = cv2torch(ground_truth)
    hsv_loss = l1_fun(predict_tensor[:, :, :], ground_truth_tensor[:, :, :])
    hsv_loss.requires_grad = True
    return hsv_loss

def labloss(predict_tensor, ground_truth_tensor):
    predict = torch2cv(predict_tensor)
    ground_truth = torch2cv(ground_truth_tensor)
    for i in range(predict.shape[0]):
        predict[i] = cv2.cvtColor(predict[i], cv2.COLOR_BGR2Lab)
        ground_truth[i] = cv2.cvtColor(ground_truth[i], cv2.COLOR_BGR2Lab)
    predict_tensor = cv2torch(predict)
    ground_truth_tensor = cv2torch(ground_truth)
    lab_loss = l2_fun(predict_tensor, ground_truth_tensor)
    lab_loss.requires_grad = True
    return lab_loss

def klloss(x, y):
    kl = F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum')
    return kl