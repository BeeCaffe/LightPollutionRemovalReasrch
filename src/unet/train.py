from src.unet.my_trainutils import *
import src.unet.valid as valid
from time import localtime, strftime
import src.unet.unet_model as unet_model
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
def logCreate(args=None):
    if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
    log_file_name = strftime('%Y-%m-%d-%H-%M-%S', localtime())+'.txt'
    log_file = open(fullfile(args.get('log_dir'), log_file_name), 'w')
    title_str = '{:<30}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}\n'
    log_file.write(title_str.format('model_name', 'loss_function', 'nums_train', 'batch_size', 'epoch',
                                    'uncmp_psnr', 'uncmp_rmse', 'uncmp_ssim', 'valid_psnr', 'valid_rmse', 'valid_ssim'))
    log_file.close()
    return log_file_name

def optionToString(train_option):
    return '{}_{}_{}_{}_{}'.format(train_option['model_name'],
                                   train_option['loss'],
                                   train_option['nums_train'],
                                   train_option['batch_size'],
                                   train_option['epoch'],)

def saveDict(args=None, save_args=None):
    if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
    log_file_name = optionToString(args)+'.npy'
    log_file = fullfile(args.get('log_dir'), log_file_name)
    np.save(log_file, save_args)

args = dict(dataset_root='datasets/dataset_512_no_gama_new/',
                loss_list=['l1+l2+ssim+vggLoss'],
                loss='',
                model_name='UNet',#pair Net self mask
                log_dir='log',
                warpping_net_pth='checkpoint/Warpping-Net_l1+l2+ssim_4000_256_50_0.001_0.2_5000_0.0001.pth',
                prospect_gamma=0.,#背景补偿更多，高光部分补偿更多
                backspect_gamma=0.,#前景，补偿更少，低光部分补偿更少
                nums_train=40,
                train_size=(256, 256),
                prj_size=(256, 256),
                gamma=2.2,
                epoch=1,
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

#log file
log_file_name = logCreate(args=args)
#load model
model = unet_model.UNet(3, 3)
if torch.cuda.device_count() >= 1: model_pro = nn.DataParallel(model, device_ids=device_ids).to(device)
#load warpping net
warping_net_model = torch.load(args['warpping_net_pth'])
#read valid data
cam_valid_path = args.get('dataset_root')+'cam/test'
prj_valid_path = args.get('dataset_root')+'prj/test'
# read valid data
cam_valid = readImgsMT(cam_valid_path, size=args['train_size'])
prj_valid = readImgsMT(prj_valid_path, size=args['prj_size'])
cam_valid = warping_net_model(cam_valid)
valid_data = dict(cam_valid=cam_valid, prj_valid=prj_valid)
for loss in args.get('loss_list'):
    args['loss'] = loss
    resetRNGseed(0)
    print('-----------------------------------------Start Train {:s}-----------------------------------'.format(args.get('model_name')))
    cam_path_dict = {
        'cam_train': 'cam/train',
        'prj_train': 'prj/train'
    }
    # back_path, pro_path, valid_psnr, valid_rmse, valid_ssim = MyTrainUtils.TrainProcess(model_pro, model_back,warping_net_model, valid_data, cam_path_dict, args).Train()
    pth, valid_psnr, valid_rmse, valid_ssim, train_msg_1 = TrainProcess(model, warping_net_model, valid_data, cam_path_dict, args).Train()
    # save result to log file
    log_file = open(fullfile(args.get('log_dir'), log_file_name), 'a')
    ret_str = '{:<30}{:<20}{:<15}{:<15}{:<15}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}\n'
    cam_valid_resize = F.interpolate(cam_valid, prj_valid.shape[2:4])
    log_file.write(ret_str.format(args.get('model_name'),
                                  loss, args['nums_train'],
                                  args['batch_size'],
                                  args['epoch'],
                                  psnr(cam_valid_resize, prj_valid),
                                  rmse(cam_valid_resize, prj_valid),
                                  ssim(cam_valid_resize, prj_valid),
                                  valid_psnr,
                                  valid_rmse,
                                  valid_ssim))
    saveDict(args, train_msg_1)
    log_file.close()
    if args['save_compensation']:
        print('------------------------------------ Start Valid {:s} ---------------------------'.format(args['model_name']))
        torch.cuda.empty_cache()
        desire_test_path = fullfile(args['dataset_root'], 'cmp/desire')
        save_path = fullfile(args['dataset_root'], 'cmp/res')
        assert os.path.isdir(desire_test_path), 'images and folder {:s} does not exist!'.format(desire_test_path)
        # compensate and save images
        valid.cmpImages(pth_path=pth, desire_test_path=desire_test_path, save_path=save_path)
        print('Compensation images saved to {}', save_path)
# clear cache
del model
torch.cuda.empty_cache()
print('----------------------------------------- Done! ------------------------------\n')


