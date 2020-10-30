from src.pairnet.TrainUtils import *
import src.pairnet.bccn_model as bccn_model
import src.pairnet.valid as valid
from time import localtime, strftime
from src.pairnet.utils import *
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

args = dict(dataset_root='datasets/seconddataset',
                loss_list=['l1+l2+ssim'],
                loss='',
                model_name='PairNet_BC',#pair Net self mask
                log_dir='log',
                test_a="test/A/",
                test_b="test/B/",
                train_a="train/A/",
                train_b="train/B/",
                test_ex_path="res/test/A/",
                test_save_path="output/",
                nums_train=2500,
                train_size=(256, 256),
                test_num=400,
                prj_size=(256, 256),
                weight_init=[1.],
                weight=0,
                epoch=10,
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
                gamma_grad=False,
                train_res_ratio=1)
#log file
log_file_name = logCreate(args=args)
polutted_valid_path = fullfile(args.get('dataset_root'), args['test_a'])
unpollted_valid_path = fullfile(args.get('dataset_root'), args['test_b'])
# read valid data
test_indexs = list(range(0, args['test_num']))
polutted_valid = readImgsMT(polutted_valid_path, size=args['train_size'], index= test_indexs)
unpolutted_valid = readImgsMT(unpollted_valid_path, size=args['prj_size'], index= test_indexs)
valid_data = dict(cam_valid=polutted_valid, prj_valid=unpolutted_valid)
for loss in args.get('loss_list'):
    for weight in args['weight_init']:
        # load model
        args['weight']=weight
        model = bccn_model.Bccn_Net(weight=args['weight'])
        if torch.cuda.device_count() >= 1: model_pro = nn.DataParallel(model, device_ids=device_ids).to(device)
        resetRNGseed(0)
        args['loss'] = loss
        print('-----------------------------------------Start Train {:s}-----------------------------------'.format(args.get('model_name')))
        data_path_dict = {
            'polluted_train': args['train_a'],
            'unpolluted_train': args['train_b']
        }
        pth, valid_psnr, valid_rmse, valid_ssim = trainModel(model, valid_data, data_path_dict, args)

        # save result to log file
        log_file = open(fullfile(args.get('log_dir'), log_file_name), 'a')
        ret_str = '{:<30}{:<20}{:<15}{:<15}{:<15}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}\n'
        cam_valid_resize = F.interpolate(polutted_valid, unpolutted_valid.shape[2:4])
        log_file.write(ret_str.format(args.get('model_name'), loss, args['nums_train'], args['batch_size'],
                                      args['epoch'], psnr(cam_valid_resize, unpolutted_valid),
                                      rmse(cam_valid_resize, unpolutted_valid),
                                      ssim(cam_valid_resize, unpolutted_valid), valid_psnr, valid_rmse, valid_ssim))
        log_file.close()
        if args['save_compensation']:
            print('------------------------------------ Start Valid {:s} ---------------------------'.format(args['model_name']))
            torch.cuda.empty_cache()
            desire_test_path = args['test_ex_path']
            save_path = args['test_save_path']
            assert os.path.isdir(desire_test_path), 'images and folder {:s} does not exist!'.format(desire_test_path)
            # compensate and save images
            valid.cmpImages(pth_path=pth, desire_test_path=desire_test_path, save_path=save_path)
            print('Compensation images saved to {}', save_path)
# clear cache
del model_pro
torch.cuda.empty_cache()
print('----------------------------------------- Done! ------------------------------\n')


