from src.unet.my_trainutils import *
import src.unet.valid as valid
import src.unet.unet_model as unet_model
import src.unet.real_unet as real_unet
import os
import torch
DEBUG = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() >= 1:
    print('Train with', torch.cuda.device_count(), 'GPU...')
else:
    print('Train with CPU...')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
args = dict(dataset_root='datasets/dataset_512_no_gama_new/',
                # loss_list=['l1','l2','l1+l2','l1+ssim','l2+ssim','l1+l2+ssim'],
                loss_list=['l1+l2+ssim+klloss'], #l1, l2, ssim, hsvloss, vggloss, labloss, klloss
                loss='',
                model_name='unet',
                res_block_num_list=[0],
                res_block_num=[],
                log_dir='log/',
                warpping_net_pth='checkpoint/Warpping-Net_l1+l2+ssim_4000_256_50_0.001_0.2_5000_0.0001.pth',
                model_type="my_unet", #"my_unet", "real_unet", "compennet"
                gamma_list=[1.2],
                nums_train=4000,
                epoch=5,
                nums_valid=50,
                train_size=(256, 256),
                prj_size=(256, 256),
                gamma=1.0,
                batch_size=16,
                vgg_lambda=1e-2,
                lr=1e-2,
                lr_drop_ratio=0.2,
                lr_drop_rate=5000,
                l2_reg=1e-4,
                device=device,
                plot_on=True,
                train_plot_rate=50,
                valid_rate=200,
                save_compensation=True,
                gamma_grad=True,
                train_res_ratio=1)

def optionToString(train_option):
    return '{}_{}_{}_{}_{}_{}_{}'.format(train_option['model_name'],
                                   train_option['loss'],
                                   train_option['nums_train'],
                                   train_option['batch_size'],
                                   train_option['epoch'],
                                    train_option['res_block_num'],
                                    train_option['gamma'])

def saveDict(args=None, save_args=None):
    log_dir = args.get('log_dir')+args['model_name']+'/'
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_file_name = optionToString(args)+'.npy'
    log_file = fullfile(log_dir, log_file_name)
    np.save(log_file, save_args)

checkpoint_pth = "checkpoint/"+args['model_name']+'/'
output_pth = "output/"+args['model_name']+'/'
desire_path = "input/unet/"
if not os.path.exists(checkpoint_pth):
    os.makedirs(checkpoint_pth)
if not os.path.exists(output_pth):
    os.makedirs(output_pth)

#load warpping net
warping_net_model = torch.load(args['warpping_net_pth'])
#read valid data
cam_valid_path = args.get('dataset_root')+'cam/test'
prj_valid_path = args.get('dataset_root')+'prj/test'
# read valid data
cam_valid = readImgsMT(cam_valid_path, size=args['train_size'], num=args['nums_valid'])
prj_valid = readImgsMT(prj_valid_path, size=args['prj_size'], num=args['nums_valid'])
cam_valid = warping_net_model(cam_valid)
valid_data = dict(cam_valid=cam_valid, prj_valid=prj_valid)
for num in args['res_block_num_list']:
    for gamma in args['gamma_list']:
        args['res_block_num'] = num
        args['gamma'] = gamma
        #log file
        #load model
        model = None
        if args['model_type']=="my_unet":
            model = unet_model.UNet(3, 3, residel_num=args['res_block_num'])
        elif args['model_type']=='real_unet':
            model = real_unet.UNet(3, 3)
        if torch.cuda.device_count() >= 1: model_pro = nn.DataParallel(model, device_ids=device_ids).to(device)
        for loss in args.get('loss_list'):
            args['loss'] = loss
            resetRNGseed(0)
            print('-----------------------------------------Start Train {:s}-----------------------------------'.format(args.get('model_name')))
            cam_path_dict = {
                'cam_train': 'cam/train',
                'prj_train': 'prj/train'
            }
            # back_path, pro_path, valid_psnr, valid_rmse, valid_ssim = MyTrainUtils.TrainProcess(model_pro, model_back,warping_net_model, valid_data, cam_path_dict, args).Train()
            pth, valid_psnr, valid_rmse, valid_ssim, train_msg_1 = TrainProcess(model,warping_net_model, valid_data, cam_path_dict, args).Train()
            # save result to log file
            saveDict(args, train_msg_1)
            if args['save_compensation']:
                print('------------------------------------ Start Valid {:s} ---------------------------'.format(args['model_name']))
                torch.cuda.empty_cache()
                assert os.path.isdir(desire_path), 'images and folder {:s} does not exist!'.format(desire_path)
                # compensate and save images
                valid.cmpImages(pth_path=pth, desire_test_path=desire_path, save_path=output_pth)
                print('Compensation images saved to {}', output_pth)
        # clear cache
        del model
        torch.cuda.empty_cache()
        print('----------------------------------------- Done! ------------------------------\n')


