import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import cv2
import src.tools.Utils  as utils

args = dict(
    dictdir='src/pairnet/log/loss/'
)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def parseDict(path):
    read_dict = np.load(path, allow_pickle=True).item()
    loss = read_dict['loss']
    rmse = read_dict['rmse'],
    valid_psnr = read_dict['valid_psnr'],
    valid_rmse = read_dict['valid_rmse'],
    valid_ssim = read_dict['valid_ssim']
    return loss, rmse, valid_psnr, valid_rmse, valid_ssim

def drawLossGraph():
    path_lists = os.listdir(args['dictdir'])
    plt.title("损失函数对比")
    loss_list = []
    name_list = []
    for dict_name in path_lists:
        path = args['dictdir']+dict_name
        loss, rmse, valid_psnr, valid_rmse, valid_ssim = parseDict(path)
        loss_list.append(loss)
        name_list.append(dict_name[:-4])

    for loss, name in zip(loss_list, name_list):
        loss = choosePoint(np.array(loss))
        plt.plot(loss)
    plt.legend(name_list)
    plt.savefig("./loss_graph.jpg")
    plt.close()
    img = cv2.imread("./loss_graph.jpg")
    return img


def drawPSNRGraph():
    plt.title("PSNR对比")
    path_lists = os.listdir(args['dictdir'])
    # plt.ylim(0, 1)
    psnr_list = []
    name_list = []
    for dict_name in path_lists:
        path = args['dictdir'] + dict_name
        loss, rmse, valid_psnr, valid_rmse, valid_ssim = parseDict(path)
        psnr_list.append(valid_psnr[0])
        name_list.append(dict_name[:-4])
    for psnr, name in zip(psnr_list, name_list):
        psnr = choosePoint(np.array(psnr))
        plt.plot(psnr)
    plt.legend(name_list)
    plt.savefig("./psnr_graph.jpg")
    plt.close()
    ret_img = cv2.imread("./psnr_graph.jpg")
    return ret_img

def drawRMSEGraph():
    plt.title("RMSE对比")
    path_lists = os.listdir(args['dictdir'])
    # plt.ylim(0, 1)
    ssim_list = []
    name_list = []
    for dict_name in path_lists:
        path = args['dictdir'] + dict_name
        loss, rmse, valid_psnr, valid_rmse, valid_ssim = parseDict(path)
        ssim_list.append(valid_rmse[0])
        name_list.append(dict_name[:-4])
    for rmse, name in zip(ssim_list, name_list):
        rmse = choosePoint(np.array(rmse))
        plt.plot(rmse)
    plt.legend(name_list)
    plt.savefig("./rmse_graph.jpg")
    plt.close()
    ret_img = cv2.imread("./rmse_graph.jpg")
    return ret_img

def drawSSIMGraph():
    plt.title("SSIM对比")
    path_lists = os.listdir(args['dictdir'])
    # plt.ylim(0, 1)
    rmse_list = []
    name_list = []
    for dict_name in path_lists:
        path = args['dictdir'] + dict_name
        loss, rmse, valid_psnr, valid_rmse, valid_ssim = parseDict(path)
        rmse_list.append(valid_rmse[0])
        name_list.append(dict_name[:-4])
    for rmse, name in zip(rmse_list, name_list):
        rmse = choosePoint(np.array(rmse))
        plt.plot(rmse)
    plt.legend(name_list)
    plt.savefig("./ssim_graph.jpg")
    plt.close()
    ret_img = cv2.imread("./ssim_graph.jpg")
    return ret_img

def choosePoint(arr, epoch_range = (0, 25000), step=500):
    new_arr = []
    start = epoch_range[0]
    end = epoch_range[1]
    for i in range(start, end, step):
        new_arr.append(arr[i])
    return np.array(new_arr)

if __name__=='__main__':
    psnr = drawPSNRGraph()
    rmse = drawRMSEGraph()
    ssim = drawSSIMGraph()
    loss = drawLossGraph()
    psnr = cv2.resize(psnr, (1920, 1080))
    rmse = cv2.resize(rmse, (1920, 1080))
    ssim = cv2.resize(ssim, (1920, 1080))
    loss = cv2.resize(loss, (1920, 1080))
    img = utils.CombineImages2D(
        [[utils.ImageAddTag(loss, "a"), utils.ImageAddTag(psnr, "b" )],
         [utils.ImageAddTag(rmse, "c" ), utils.ImageAddTag(ssim, "d" )]]
    )
    cv2.imwrite("./img.jpg", img)