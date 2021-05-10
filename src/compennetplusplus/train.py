from src.compennetplusplus.trainNetwork import *
import src.compennetplusplus.CompenNetPlusplusModel as CompenNetPlusplusModel
from time import localtime, strftime
import src.compennetplusplus.valid as valid
import src.compennetplusplus.ImgProc as ImgProc
import src.compennetplusplus.utils as utils
import numpy as np

import os
import torch
import cv2 as cv
__DEBUG=False
os.environ['CUDA_VISIBLE_DEVICES']='0'
device_ids=[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()>=1:
    print('Train with',torch.cuda.device_count(),'GPU...')
else:
    print('Train with CPU...')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
args = {
    'dataset_root': 'data/dataset1',
    # 'loss_list': ['l2+ssim', 'l1+l2+ssim','l1', 'l2', 'ssim', 'l1+l2'],
    'loss_list': ['l1+ssim'],
    'loss': '',#it would be set later
    'model_name': 'CompenNet++',
    'log_dir': 'log',
    'nums_train': 4000, #how many images you suppose to train ,if 0 it will train all images
    'train_size': (1080, 1920), #the images train size mush equal to output image size.for
    'epoch': 10,
    'batch_size': 1,
    'lr': 1e-3,
    'lr_drop_ratio': 0.2,
    'lr_drop_rate': 5000,
    'l2_reg': 1e-4,
    'device': device,
    'plot_on': True,
    'train_plot_rate': 50,
    'valid_rate': 200,
    'save_compensation': True,
    'mask_corner': None,#mask corner would set later
    'prj_fov_mask': None,#projector mask would be set later
}
#log file
if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
log_file_name = strftime('%Y-%m-%d-%H-%M-%S', localtime())+'.txt'
log_file = open(utils.fullfile(args.get('log_dir'), log_file_name),'w')
title_str = '{:<30}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}\n'
log_file.write(title_str.format('model_name', 'loss_function','nums_train','batch_size', 'epoch',
                                'uncmp_psnr', 'uncmp_rmse','uncmp_ssim','valid_psnr','valid_rmse','valid_ssim'))
log_file.close()

#load model
compen_net_model = CompenNetPlusplusModel.CompenNet()
if torch.cuda.device_count() >= 1: compen_net_model = nn.DataParallel(compen_net_model, device_ids=device_ids).to(device)
compen_net_model=initCompenNet(compen_net_model, args.get('dataset_root'),args.get('device'))

#get mask corners
cam_ref_path = utils.fullfile(args.get('dataset_root'), 'cam/ref')
mask_corner, prj_fov_mask=ImgProc.get_mask_corner(cam_ref_path=cam_ref_path,train_option=args)
args['mask_corner']=mask_corner
args['prj_fov_mask']=prj_fov_mask
if __DEBUG:
    pass
    # img = prj_fov_mask.permute(0,2,3,1).mul(255).numpy()
    # img = np.uint8(img)
    # cv.imshow('debug img', img[0])
    # cv.waitKey(0)

#read valid data
cam_surf_path = utils.fullfile(args.get('dataset_root'), 'cam/ref')
cam_valid_path = utils.fullfile(args.get('dataset_root'), 'cam/test')
prj_valid_path = utils.fullfile(args.get('dataset_root'), 'prj/test')
# read valid data
cam_valid = utils.readImgsMT(cam_valid_path, size=args['train_size'])
prj_valid = utils.readImgsMT(prj_valid_path, size=args['train_size'])
cam_surf_valid = utils.readImgsMT(cam_surf_path, size=args.get('train_size'), index=[0])
cam_surf_valid[~prj_fov_mask] = 0
cam_surf_valid=cam_surf_valid.expand_as(cam_valid)
valid_data = dict(cam_surf=cam_surf_valid, cam_valid=cam_valid, prj_valid=prj_valid)


for loss in args.get('loss_list'):
    resetRNGseed(0)
    warping_net_model=CompenNetPlusplusModel.WarpingNet(out_size=args.get('train_size'))
    src_pts = np.array([[-1, -1], [1, -1], [1, 1]]).astype(np.float32)
    dst_pts = np.array(mask_corner[0][0:3]).astype(np.float32)
    affine_mat = cv.getAffineTransform(src_pts, dst_pts)
    warping_net_model.set_affine(affine_mat.flatten())
    if torch.cuda.device_count() >= 1: warping_net_model = nn.DataParallel(warping_net_model, device_ids=device_ids).to(device)
    compen_net_pp_model=CompenNetPlusplusModel.CompenNetPlusplus(warping_net_model, compen_net_model)
    if torch.cuda.device_count() >= 1: compen_net_pp_model = nn.DataParallel(compen_net_pp_model, device_ids=device_ids).to(device)
    args['loss']=loss
    print('-----------------------------------------Start Train {:s}-----------------------------------'.format(args.get('model_name')))
    pth_path, valid_psnr, valid_rmse,valid_ssim = trainModel(compen_net_pp_model,valid_data,args)
    #save result to log file
    log_file = open(fullfile(args.get('log_dir'), log_file_name), 'a')
    ret_str = '{:<30}{:<20}{:<15}{:<15}{:<15}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}\n'
    cam_valid_resize = F.interpolate(cam_valid, prj_valid.shape[2:4])
    log_file.write(ret_str.format(args.get('model_name'), loss, args['nums_train'], args['batch_size'],
                                  args['epoch'], psnr(cam_valid_resize,prj_valid), rmse(cam_valid_resize, prj_valid),
                                  ssim(cam_valid_resize, prj_valid), valid_psnr, valid_rmse, valid_ssim))
    log_file.close()
    if args['save_compensation']:
        print('------------------------------------ Start testing {:s} ---------------------------'.format(args['model_name']))
        torch.cuda.empty_cache()
        # desired test images are created such that they can fill the optimal displayable area (see paper for detail)
        desire_test_path = utils.fullfile(args['dataset_root'], 'cmp/desire')
        cmp_surf_path = utils.fullfile(args['dataset_root'], 'cmp/surf/surf.jpg')
        save_path = utils.fullfile(args['dataset_root'], 'cmp/res')
        assert os.path.isdir(desire_test_path), 'images and folder {:s} does not exist!'.format(desire_test_path)
        # compensate and save images
        valid.cmpImages(pth_path=pth_path, desire_test_path=desire_test_path, cmp_surf_path=cmp_surf_path, save_path=save_path)
        print('Compensation images saved to {}', save_path)
        # clear cache
        del compen_net_pp_model, warping_net_model
        torch.cuda.empty_cache()
        print('----------------------------------------- Done! ------------------------------\n')
