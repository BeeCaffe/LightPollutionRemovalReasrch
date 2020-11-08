from src.wrapnet.trainUtils import *
import src.wrapnet.CompenNetPlusplusModel as CompenNetPlusplusModel
from time import localtime, strftime
import src.wrapnet.ImgProc as ImgProc
import src.wrapnet.valid as valid
DEBUG = False
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device_ids = [0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() >= 1:
    print('Train with', torch.cuda.device_count(), 'GPU...')
else:
    print('Train with CPU...')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
args = {
    'dataset_root': 'datasets/geocor/',
    'loss_list': ['l1+l2+ssim'],
    'loss': '',#it would be set later---
    'lambda': 1e-2,
    'model_name': 'Warpping-Net-1920-1080',
    'log_dir': 'log',
    'nums_train': 50, #how many images you suppose to train ,if 0 it will train all images
    'train_size': (1080, 1920),
    'prj_size': (1080, 1920),
    'epoch': 100,
    'batch_size': 4,
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
def evaluateWarpingNet(model, valid_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cam_surf = valid_data['cam_surf']
    cam_valid = valid_data['cam_valid']
    prj_valid = valid_data['prj_valid']

    with torch.no_grad():
        model.eval()  # explicitly set to eval mode

        # if you have limited GPU memory, we need to predict in batch mode
        if cam_surf.device.type != device.type:
            last_loc = 0
            valid_mse, valid_ssim = 0., 0.

            prj_valid_pred = torch.zeros(prj_valid.shape)
            num_valid = cam_valid.shape[0]
            batch_size = 50 if num_valid > 50 else num_valid  # default number of test images per dataset

            for i in range(0, num_valid // batch_size):
                idx = range(last_loc, last_loc + batch_size)
                cam_surf_batch = cam_surf[idx, :, :, :].to(device) if cam_surf.device.type != 'cuda' else cam_surf[idx, :, :, :]
                cam_valid_batch = cam_valid[idx, :, :, :].to(device) if cam_valid.device.type != 'cuda' else cam_valid[idx, :, :, :]
                prj_valid_batch = prj_valid[idx, :, :, :].to(device) if prj_valid.device.type != 'cuda' else prj_valid[idx, :, :, :]

                # predict batch
                # prj_valid_pred_batch = predict(model, dict(cam=cam_valid_batch, cam_surf=cam_surf_batch)).detach()
                prj_valid_pred_batch = model(cam_valid_batch).detach()
                if type(prj_valid_pred_batch) == tuple and len(prj_valid_pred_batch) > 1: prj_valid_pred_batch = prj_valid_pred_batch[0]
                prj_valid_pred[last_loc:last_loc + batch_size, :, :, :] = prj_valid_pred_batch.cpu()

                # compute loss
                valid_mse += l2_fun(prj_valid_pred_batch, prj_valid_batch).item() * batch_size
                valid_ssim += ssim(prj_valid_pred_batch, prj_valid_batch) * batch_size

                last_loc += batch_size
            # average
            valid_mse /= num_valid
            valid_ssim /= num_valid

            valid_rmse = math.sqrt(valid_mse * 3)  # 3 channel image
            valid_psnr = 10 * math.log10(1 / valid_mse)
        else:
            # if all data can be loaded to GPU memory
            prj_valid_pred = predict(model, dict(cam=cam_valid, cam_surf=cam_surf)).detach()
            valid_mse = l2_fun(prj_valid_pred, prj_valid).item()
            valid_rmse = math.sqrt(valid_mse * 3)  # 3 channel image
            valid_psnr = 10 * math.log10(1 / valid_mse)
            valid_ssim = ssim_fun(prj_valid_pred, prj_valid).item()

    return valid_psnr, valid_rmse, valid_ssim, prj_valid_pred

def trainWarpingNet(model, valid_data,cam_path_dict,train_option,is_compenNet=False):
    '''
    :param model: compenNet++ model
    :param train_list: contains 'cam_surf (size 1)', 'cam_train (size n)' and 'prj_train (size n)' path lists
    :param valid_list: contains 'cam_valid (size n)', ''prj_valid (size n)' path lists
    :param train_option: 'args' in train.py
    :return:
    '''
    device = train_option.get('device')
    if device.type == 'cuda':torch.cuda.empty_cache()
    #list of parameters to be optimized
    params = filter(lambda param:param.requires_grad, model.parameters())
    #optimizer
    optimizer = optim.Adam(params, lr=train_option.get('lr'), weight_decay=train_option.get('l2_reg'))
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_option.get('lr_drop_rate'),gamma=train_option.get('lr_drop_ratio'))
    start_time = time.time()
    #train data path
    cam_surf_path = fullfile(train_option.get('dataset_root'), cam_path_dict['cam_surf'])
    cam_train_path = fullfile(train_option.get('dataset_root'), cam_path_dict['cam_train'])
    prj_train_path = fullfile(train_option.get('dataset_root'), cam_path_dict['prj_train'])
    valid_psnr, valid_rmse, valid_ssim = 0., 0., 0.
    for i in range(train_option['epoch']-1):
        iters = 0
        while iters < train_option.get('nums_train')//train_option['batch_size']:
            # idx = random.sample(range(train_option.get('nums_train')), train_option.get('batch_size'))
            idx = [j for j in range(iters*train_option['batch_size'], (iters+1)*train_option['batch_size'])]
            cam_train_batch = loadDataByPaths(cam_train_path, idx, size=train_option['train_size'], prj_fov_mask=train_option['prj_fov_mask']).to(device)
            if is_compenNet == False:
                if DEBUG:
                    img = cam_train_batch[0].permute(1, 2, 0).mul(255).to('cpu').detach().numpy()
                    img = np.uint8(img)
                    cv2.imshow('cam train img unmasked'
                               '', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                cam_train_batch[~train_option['prj_fov_mask'].expand_as(cam_train_batch)] = 0
                if DEBUG:
                    img = cam_train_batch[0].permute(1, 2, 0).mul(255).to('cpu').detach().numpy()
                    img = np.uint8(img)
                    cv2.imshow('cam train img masked'
                               '', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                prj_train_batch = loadDataByPaths(prj_train_path, idx=idx, size=train_option['train_size']).to(device)
                cam_surf_train_batch = readImgsMT(cam_surf_path, size=train_option.get('train_size'), index=[0])
            else:
                prj_train_batch = loadDataByPaths(prj_train_path, idx=idx, size=train_option['train_size']).to(device)
                cam_surf_train_batch = readImgsMT(cam_surf_path, size=train_option.get('train_size'), index=[0])
            if is_compenNet == False:
                cam_surf_train_batch[~train_option['prj_fov_mask']] = 0
                cam_surf_train_batch = cam_surf_train_batch.to(device).expand_as(cam_train_batch)
            else:
                cam_surf_train_batch = cam_surf_train_batch.to(device).expand_as(cam_train_batch)
            #predict and compute loss
            model.train()
            if DEBUG:
                img = cam_train_batch[0].permute(1, 2, 0).mul(255).to('cpu').detach().numpy()
                img = np.uint8(img)
                cv2.imshow('cam_train_batch'
                           '', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            prj_train_pred = model(cam_train_batch)
            if DEBUG:
                img = prj_train_batch[0].permute(1, 2, 0).mul(255).to('cpu').detach().numpy()
                img = np.uint8(img)
                cv2.imshow('prj_train_batch'
                           '', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            train_loss_batch, train_l2_loss_batch = computeLoss(prj_train_pred, prj_train_batch, train_option.get('loss'))


            # perceptual loss
            if 'vggLoss' in train_option['loss']:
                predictionBatch = model(prj_train_batch, cam_surf_train_batch)
                perceptLossBatch = pLoss.perceptualLoss(predictionBatch, prj_train_batch)
                train_loss_batch = train_loss_batch + train_l2_loss_batch + train_option['lambda'] * perceptLossBatch
            train_rmse_batch = math.sqrt(train_l2_loss_batch.item() * 3)  # 3 channels rgb

            #backpropagation and update params
            optimizer.zero_grad()
            train_loss_batch.backward()
            optimizer.step()
            #record running time
            time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
            del cam_train_batch, cam_surf_train_batch, prj_train_batch
            lr_scheduler.step()
            iters += 1
            #plot train
            valid_psnr, valid_rmse, valid_ssim = 0., 0., 0.
            #validation
            if valid_data is not None:
                valid_psnr, valid_rmse, valid_ssim, prj_valid_pred = evaluateWarpingNet(model, valid_data=valid_data)
            print('Epoch:{:5d}|Iter:{:5d}|Time:{}|Train Loss:{:.4f}|Train RMSE: {:.4f}|Valid PSNR: {:7s}|Valid RMSE: {:6s}'
                      '|Valid SSIM: {:6s}|Learn Rate: {:.5f}'.format(i, iters, time_lapse, train_loss_batch.item(), train_rmse_batch,
                                                                    '{:>2.4f}'.format(valid_psnr) if valid_psnr else '',
                                                                    '{:.4f}'.format(valid_rmse) if valid_rmse else '',
                                                                    '{:.4f}'.format(valid_ssim) if valid_ssim else '',
                                                                    optimizer.param_groups[0]['lr']))

    #Done training and save the last epoch model
    checkpoint_dir = 'checkpoint'
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    title = optionToString(train_option)
    checkpoint_file_name = fullfile(checkpoint_dir, title+'.pth')
    torch.save(model, checkpoint_file_name)
    print('Checkpoint saved to {}\n'.format(checkpoint_file_name))
    return checkpoint_file_name, valid_psnr, valid_rmse, valid_ssim,model


#log file
if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
log_file_name = strftime('%Y-%m-%d-%H-%M-%S', localtime())+'.txt'
log_file = open(fullfile(args.get('log_dir'), log_file_name),'w')
title_str = '{:<30}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}\n'
log_file.write(title_str.format('model_name', 'loss_function','nums_train','batch_size', 'epoch',
                                'uncmp_psnr', 'uncmp_rmse','uncmp_ssim','valid_psnr','valid_rmse','valid_ssim'))
log_file.close()


#get mask corners
cam_ref_path = fullfile(args.get('dataset_root'), 'cam/ref')
mask_corner, prj_fov_mask = ImgProc.get_mask_corner(cam_ref_path=cam_ref_path,train_option=args)
# print(mask_corner)
args['mask_corner'] = mask_corner
args['prj_fov_mask'] = prj_fov_mask
np.savetxt("./mask_corner.txt", mask_corner[0])
if DEBUG:
    pass
    # img = prj_fov_mask.permute(0,2,3,1).mul(255).numpy()
    # img = np.uint8(img)
    # cv.imshow('debug img', img[0])
    # cv.waitKey(0)

#read valid data
cam_surf_path = fullfile(args.get('dataset_root'), 'cam/ref')
cam_valid_path = fullfile(args.get('dataset_root'), 'cam/test')
prj_valid_path = fullfile(args.get('dataset_root'), 'prj/test')
# read valid data
cam_valid = readImgsMT(cam_valid_path, size=args['train_size'])
prj_valid = readImgsMT(prj_valid_path, size=args['train_size'])
cam_surf_valid = readImgsMT(cam_surf_path, size=args.get('train_size'), index=[0])
cam_surf_valid[~prj_fov_mask] = 0
cam_surf_valid = cam_surf_valid.expand_as(cam_valid)
valid_data = dict(cam_surf=cam_surf_valid, cam_valid=cam_valid, prj_valid=prj_valid)


for loss in args.get('loss_list'):
    resetRNGseed(0)
    warping_net_model=CompenNetPlusplusModel.WarpingNet(out_size=args.get('train_size'))
    src_pts = np.array([[-1, -1], [1, -1], [1, 1]]).astype(np.float32)
    dst_pts = np.array(mask_corner[0][0:3]).astype(np.float32)
    affine_mat = cv.getAffineTransform(src_pts, dst_pts)
    warping_net_model.set_affine(affine_mat.flatten())
    if torch.cuda.device_count() >= 1: warping_net_model = nn.DataParallel(warping_net_model, device_ids=device_ids).to(device)
    args['loss']=loss
    print('-----------------------------------------Start Train {:s}-----------------------------------'.format(args.get('model_name')))
    cam_path_dict = {
        'cam_surf': 'cam/ref',
        'cam_train': 'cam/train',
        'prj_train': 'prj/train'
    }
    pth_path, valid_psnr, valid_rmse,valid_ssim,_=trainWarpingNet(warping_net_model, valid_data, cam_path_dict, args, is_compenNet=False)
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
        desire_test_path = fullfile(args['dataset_root'], 'cam/test')
        cmp_surf_path = fullfile(args['dataset_root'], 'cmp/surf/surf.jpg')
        save_path = fullfile(args['dataset_root'], 'cmp/res')
        assert os.path.isdir(desire_test_path), 'images and folder {:s} does not exist!'.format(desire_test_path)
        # compensate and save images
        valid.cmpImages(pth_path=pth_path, desire_test_path=desire_test_path, save_path=save_path)
        print('Compensation images saved to {}', save_path)
        # clear cache
        del  warping_net_model
        torch.cuda.empty_cache()
        print('----------------------------------------- Done! ------------------------------\n')
