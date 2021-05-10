'''
CompenNet++ training functions
'''

from src.compennetplusplus.utils import *
import torch.nn.functional as F
import torch.optim as optim
import time
import visdom
import src.compennetplusplus.ImgProc as ImgProc
import src.compennetplusplus.myutils as myutils
import src.compennetplusplus.pytorch_ssim as pytorch_ssim
import cv2

__DEBUG = False

# for visualization
vis = visdom.Visdom()
assert vis.check_connection(), 'Visdom: No connection, start visdom first!'

l1_fun = nn.L1Loss()
l2_fun = nn.MSELoss()
ssim_fun = pytorch_ssim.SSIM()
def trainModel(model, valid_data,train_option):
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
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=train_option.get('lr_drop_rate'),gamma=train_option.get('lr_drop_ratio'))
    start_time=time.time()
    #train data path
    cam_surf_path = fullfile(train_option.get('dataset_root'), 'cam/ref')
    cam_train_path = fullfile(train_option.get('dataset_root'), 'cam/train')
    prj_train_path = fullfile(train_option.get('dataset_root'), 'prj/train')
    valid_psnr, valid_rmse, valid_ssim = 0., 0., 0.
    for i in range(train_option['epoch']-1):
        iters = 0
        print("numbers train:{}".format(train_option.get('nums_train')))
        while iters < train_option.get('nums_train')//train_option['batch_size']:
            # idx = random.sample(range(train_option.get('nums_train')), train_option.get('batch_size'))
            idx = [j for j in range(iters*train_option['batch_size'], (iters+1)*train_option['batch_size'])]
            cam_train_batch = loadDataByPaths(cam_train_path, idx, train_option, cam=False).to(device)
            cam_train_batch[~train_option['prj_fov_mask'].expand_as(cam_train_batch)] = 0
            prj_train_batch = loadDataByPaths(prj_train_path, idx, train_option).to(device)
            cam_surf_train_batch = readImgsMT(cam_surf_path, size=train_option.get('train_size'), index=[0])
            cam_surf_train_batch[~train_option['prj_fov_mask']]=0
            if __DEBUG:
                img = cam_surf_train_batch[0].permute(1, 2, 0).mul(255).to('cpu').numpy()
                img = np.uint8(img)
                cv2.imshow('debug img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            cam_surf_train_batch = cam_surf_train_batch.to(device).expand_as(prj_train_batch)
            #predict and compute loss
            model.train()
            prj_train_pred = predict(model, dict(cam=cam_train_batch, cam_surf=cam_surf_train_batch))
            train_loss_batch, train_l2_loss_batch = computeLoss(prj_train_pred, prj_train_batch, train_option.get('loss'))
            train_rmse_batch = math.sqrt(train_l2_loss_batch.item()*3) #3 channels rgb
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
                valid_psnr, valid_rmse, valid_ssim, prj_valid_pred = evaluate(model, valid_data=valid_data)
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
    return checkpoint_file_name, valid_psnr, valid_rmse, valid_ssim

def loadDataByPaths(datapath, idx=None, args=None,cam=False):
    if args is None:
        raise RuntimeError("args dict should not be None")
    if idx is not None:
        data = readImgsMT(datapath, size=args.get('train_size'), index=idx)
    else:
        data = readImgsMT(datapath, size=args.get('train_size'))
    if cam:
        data = torch.where(args.get('prj_fov_mask'), data, torch.tensor([0.]))
    return data

# initialize CompenNet to |x-s| without actual projections
def initCompenNet(compen_net, dataset_root, device):
    ckpt_file = 'checkpoint/init_CompenNet_l1+ssim_5000_8_2000_0.001_0.2_5000_0.0001.pth'

    if os.path.exists(ckpt_file):
        # load weights initialized CompenNet from saved state dict
        compen_net.load_state_dict(torch.load(ckpt_file))

        print('CompenNet state dict found! Loading...')
    else:
        # then initialize compenNet to |x-s|
        init_option = {
            'dataset_root': 'data/dataset1/init',
            'loss': 'l1+ssim',  # it would be set later
            'data_name': 'init',
            'model_name': 'init_CompenNet',
            'log_dir': 'log',
            'nums_train': 5000,
            'num_train': 400,
            'train_size': (256, 256),
            'max_epochs': 2000,
            'batch_size': 8,
            'max_iters': 5000,
            'epoch': 2000,
            'lr': 1e-3,
            'lr_drop_ratio': 0.2,
            'lr_drop_rate': 5000,
            'l2_reg': 1e-4,
            'device': device,
            'plot_on': True,
            'train_plot_rate': 50,
            'valid_rate': 200,
            'save_compensation': True,
            'mask_corner': None,  # mask corner would set later
            'prj_fov_mask': None,  # projector mask would be set later
        }
        # load data
        prj_train = readImgsMT('data/dataset1/init/prj/train')
        cam_surf = readImgsMT('data/dataset1/init/cam/ref')
        init_data = dict(cam_surf=cam_surf.expand_as(prj_train),
                         cam_train=cam_surf.expand_as(prj_train),
                         prj_train=prj_train)
        compen_net, _, _, _ = trainModelOld(compen_net, init_data, None, train_option=init_option)
    return compen_net


def trainModelOld(model, train_data, valid_data, train_option):
    device = train_option['device']

    # empty cuda cache before training
    if device.type == 'cuda': torch.cuda.empty_cache()

    # training data
    cam_surf_train = train_data['cam_surf']
    cam_train = train_data['cam_train']
    prj_train = train_data['prj_train']

    # list of parameters to be optimized
    params = filter(lambda param: param.requires_grad, model.parameters())  # only optimize parameters that require gradient
    # optimizer
    optimizer = optim.Adam(params, lr=train_option['lr'], weight_decay=train_option['l2_reg'])
    # learning rate drop scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_option['lr_drop_rate'], gamma=train_option['lr_drop_ratio'])
    # %% start train
    start_time = time.time()
    # get model name
    if not 'model_name' in train_option: train_option['model_name'] = model.name if hasattr(model, 'name') else model.module.name
    # initialize visdom data visualization figure
    if 'plot_on' not in train_option: train_option['plot_on'] = True
    if train_option['plot_on']:
        # title string of current training option
        title = optionToString(train_option)

        # intialize visdom figures
        vis_train_fig = None
        vis_valid_fig = None
        vis_curve_fig = vis.line(X=np.array([0]), Y=np.array([0]),
                                 opts=dict(title=title, font=dict(size=20), layoutopts=dict(
                                     plotly=dict(xaxis={'title': 'Iteration'}, yaxis={'title': 'Metrics', 'hoverformat': '.4f'})),
                                           width=1300, height=500, markers=True, markersize=3),
                                 name='origin')

    # main loop
    iters = 0

    while iters < train_option['max_iters']:
        # randomly sample training batch and send to GPU
        idx = random.sample(range(train_option['num_train']), train_option['batch_size'])
        cam_surf_train_batch = cam_surf_train[idx, :, :, :].to(device) if cam_surf_train.device.type != 'cuda' else cam_surf_train[idx, :, :, :]
        cam_train_batch = cam_train[idx, :, :, :].to(device) if cam_train.device.type != 'cuda' else cam_train[idx, :, :, :]
        prj_train_batch = prj_train[idx, :, :, :].to(device) if prj_train.device.type != 'cuda' else prj_train[idx, :, :, :]
        # predict and compute loss
        model.train()  # explicitly set to train mode in case batchNormalization and dropout are used
        prj_train_pred = predict(model, dict(cam=cam_train_batch, cam_surf=cam_surf_train_batch))
        train_loss_batch, train_l2_loss_batch = computeLoss(prj_train_pred, prj_train_batch, train_option['loss'])
        train_rmse_batch = math.sqrt(train_l2_loss_batch.item() * 3)  # 3 channels, rgb
        # backpropagation and update params
        optimizer.zero_grad()
        train_loss_batch.backward()
        optimizer.step()
        # record running time
        time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        # plot train
        if train_option['plot_on']:
            if iters % train_option['train_plot_rate'] == 0 or iters == train_option['max_iters'] - 1:
                vis_train_fig = plotMontage(dict(cam=cam_train_batch.detach(), pred=prj_train_pred.detach(), prj=prj_train_batch.detach()),
                                            win=vis_train_fig, title='[Train]' + title)
                appendDataPoint(iters, train_loss_batch.item(), vis_curve_fig, 'train_loss')
                appendDataPoint(iters, train_rmse_batch, vis_curve_fig, 'train_rmse')
        # validation
        valid_psnr, valid_rmse, valid_ssim = 0., 0., 0.
        if valid_data is not None and (iters % train_option['valid_rate'] == 0 or iters == train_option['max_iters'] - 1):
            valid_psnr, valid_rmse, valid_ssim, prj_valid_pred = evaluate(model, valid_data)
            # plot validation
            if train_option['plot_on']:
                vis_valid_fig = plotMontage(dict(cam=valid_data['cam_valid'], pred=prj_valid_pred, prj=valid_data['prj_valid']),
                                            win=vis_valid_fig, title='[Valid]' + title)
                appendDataPoint(iters, valid_rmse, vis_curve_fig, 'valid_rmse')
                appendDataPoint(iters, valid_ssim, vis_curve_fig, 'valid_ssim')
        # print to console
        print('Iter:{:5d} | Time: {} | Train Loss: {:.4f} | Train RMSE: {:.4f} | Valid PSNR: {:7s}  | Valid RMSE: {:6s}  '
              '| Valid SSIM: {:6s}  | Learn Rate: {:.5f} |'.format(iters, time_lapse, train_loss_batch.item(), train_rmse_batch,
                                                                   '{:>2.4f}'.format(valid_psnr) if valid_psnr else '',
                                                                   '{:.4f}'.format(valid_rmse) if valid_rmse else '',
                                                                   '{:.4f}'.format(valid_ssim) if valid_ssim else '',
                                                                   optimizer.param_groups[0]['lr']))
        lr_scheduler.step()  # update learning rate according to schedule
        iters += 1
    # Done training and save the last epoch model
    checkpoint_dir = 'checkpoint'
    if not os.path.exists(checkpoint_dir) : os.makedirs(checkpoint_dir)
    checkpoint_file_name = fullfile(checkpoint_dir, title + '.pth')
    torch.save(model.state_dict(), checkpoint_file_name)
    print('Checkpoint saved to {}\n'.format(checkpoint_file_name))
    print(model)
    return model, valid_psnr, valid_rmse, valid_ssim

# compute loss between prediction and ground truth
def computeLoss(prj_pred, prj_train, loss_option):
    # l1
    l1_loss = 0
    if 'l1' in loss_option:
        l1_loss = l1_fun(prj_pred, prj_train)

    # l2
    l2_loss = l2_fun(prj_pred, prj_train)

    # ssim
    ssim_loss = 0
    if 'ssim' in loss_option:
        ssim_loss = 1 * (1 - ssim_fun(prj_pred, prj_train))

    train_loss = 0
    # linear combination of losses
    if loss_option == 'l1':
        train_loss = l1_loss
    elif loss_option == 'l2':
        train_loss = l2_loss
    elif loss_option == 'l1+l2':
        train_loss = l1_loss + l2_loss
    elif loss_option == 'ssim':
        train_loss = ssim_loss
    elif loss_option == 'l1+ssim':
        train_loss = l1_loss + ssim_loss
    elif loss_option == 'l2+ssim':
        train_loss = l2_loss + ssim_loss
    elif loss_option == 'l1+l2+ssim':
        train_loss = l1_loss + l2_loss + ssim_loss
    else:
        print('Unsupported loss')

    return train_loss, l2_loss

# plot sample predicted images using visdom
def plotMontage(images, win=None, title=None, env=None):
    cam_im = images['cam']  # Camera catpured uncompensated images
    prj_pred = images['pred']  # CompenNet predicted projector input images
    prj_im = images['prj']  # Ground truth of projector input images

    if cam_im.device.type == 'cpu' or prj_pred.device.type == 'cpu' or prj_im.device.type == 'cpu':
        cam_im = cam_im.cpu()
        prj_pred = prj_pred.cpu()
        prj_im = prj_im.cpu()

    # compute montage grid size
    # step = 1
    if cam_im.shape[0] > 5:
        grid_w = 5
        # step = round(cam_im.shape[0] / grid_w)
        # grid_w = len(range(0, cam_im.shape[0], step))
        idx = random.sample(range(0, cam_im.shape[0]), grid_w)
    else:
        grid_w = cam_im.shape[0]
        idx = random.sample(range(0, cam_im.shape[0]), grid_w)
    # resize if the image sizes are not the same
    # if cam_im.shape != prj_im.shape:
    #     cam_im_resize = F.interpolate(cam_im[::step, :, :, :], (prj_im.shape[2:4]))
    # else:
    #     cam_im_resize = cam_im[::step, :, :, :]

    # resize to (256, 256) for better display
    tile_size = (256, 256)
    if cam_im.shape[2] != tile_size[0] or cam_im.shape[3] != tile_size[1]:
        cam_im_resize = F.interpolate(cam_im[idx, :, :, :], tile_size)
    else:
        cam_im_resize = cam_im[idx, :, :, :]
    if prj_im.shape[2] != tile_size[0] or prj_im.shape[3] != tile_size[1]:
        prj_im_resize = F.interpolate(prj_im[idx, :, :, :], tile_size)
    else:
        prj_im_resize = prj_im[idx, :, :, :]
    if prj_pred.shape[2] != tile_size[0] or prj_pred.shape[3] != tile_size[1]:
        prj_pred_resize = F.interpolate(prj_pred[idx, :, :, :], tile_size)
    else:
        prj_pred_resize = prj_pred[idx, :, :, :]

    # % view results
    im_concat = torch.cat((cam_im_resize, prj_pred_resize, prj_im_resize), 0)
    im_montage = montage(im_concat, multichannel=True, padding_width=10, fill=[1, 1, 1], grid_shape=(3, grid_w))

    # title
    plot_opts = dict(title=title, caption=title, font=dict(size=18))

    # plot montage to existing win, otherwise create a new win
    # win = vis.image(im_montage.transpose(2, 0, 1), win=win, opts=plot_opts, env=env)
    return win


# predict projector input images given input data (do not use with torch.no_grad() within this function)
def predict(model, data):
    if 'cam_surf' in data and data['cam_surf'] is not None:
        prj_pred = model(data['cam'], data['cam_surf'])
    else:
        prj_pred = model(data['cam'])

    if type(prj_pred) == tuple and len(prj_pred) > 1: prj_pred = prj_pred[0]
    return prj_pred


# evaluate model on validation dataset
def evaluate(model, valid_data):
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
                prj_valid_pred_batch = predict(model, dict(cam=cam_valid_batch, cam_surf=cam_surf_batch)).detach()
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

# append a data point to the curve in win
def appendDataPoint(x, y, win, name, env=None):
    vis.line(
        X=np.array([x]),
        Y=np.array([y]),
        env=env,
        win=win,
        update='append',
        name=name,
        opts=dict(markers=True, markersize=3)
    )

# generate training title string
def optionToString(train_option):
    return '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(train_option['model_name'], train_option['loss'],
                                                  train_option['nums_train'], train_option['batch_size'], train_option['epoch'],
                                                  train_option['lr'], train_option['lr_drop_ratio'], train_option['lr_drop_rate'],
                                                  train_option['l2_reg'])
