import numpy as np
import pickle
args = dict(
    dictdir='log/UNet_l1+l2+ssim+vggLoss_40_16_1.npy'
)

read_dict = np.load(args['dictdir'], allow_pickle=True).item()

def parseDict():
    loss = read_dict['loss']
    rmse = read_dict['rmse'],
    valid_psnr = read_dict['valid_psnr'],
    valid_rmse = read_dict['valid_rmse'],
    valid_ssim = read_dict['valid_ssim']
    return loss, rmse, valid_psnr, valid_rmse, valid_ssim

if __name__=='__main__':
    parseDict()