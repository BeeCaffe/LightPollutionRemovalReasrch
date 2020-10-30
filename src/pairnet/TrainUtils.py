from src.pairnet.utils import *
import torch.nn.functional as F
import torch.optim as optim
import time
import visdom
import src.pairnet.pytorch_ssim as pLoss
import src.pairnet.PerceptualLoss as perceptual
DEBUG = True
import cv2
# for visualization
vis = visdom.Visdom()
assert vis.check_connection(), 'Visdom: No connection, start visdom first!'

l1_fun = nn.L1Loss()
l2_fun = nn.MSELoss()
ssim_fun = pLoss.SSIM()

def trainModel(model, valid_data, train_path_dict, train_option):
    device = train_option.get('device')
    if device.type == 'cuda': torch.cuda.empty_cache()
    #list of parameters to be optimized
    params = filter(lambda param: param.requires_grad, model.parameters())
    #optimizer
    optimizer = optim.Adam(params, lr=train_option.get('lr'), weight_decay=train_option.get('l2_reg'))
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_option.get('lr_drop_rate'), gamma=train_option.get('lr_drop_ratio'))
    start_time = time.time()
    #train data path
    cam_train_path = fullfile(train_option.get('dataset_root'), train_path_dict['polluted_train'])
    prj_train_path = fullfile(train_option.get('dataset_root'), train_path_dict['unpolluted_train'])
    valid_psnr, valid_rmse, valid_ssim = 0., 0., 0.
    for i in range(train_option['epoch']):
        iters = 0
        while iters < train_option.get('nums_train')//train_option['batch_size']:
            idx = [j for j in range(iters*train_option['batch_size'], (iters+1)*train_option['batch_size'])]
            polluted_train_batch = loadDataByPaths(cam_train_path, idx, size=train_option['train_size']).to(device)
            unpolluted_train_batch = loadDataByPaths(prj_train_path, idx=idx, size=train_option['train_size']).to(device)
            model.train()
            polluted_train_pred = model(polluted_train_batch, unpolluted_train_batch)
            train_loss_batch, train_l2_loss_batch = computeLoss(polluted_train_pred, unpolluted_train_batch, train_option.get('loss'))
            # perceptual loss
            if 'vggLoss' in train_option['loss']:
                perceptLossBatch = perceptual.perceptualLoss(polluted_train_pred, unpolluted_train_batch)
                train_loss_batch = train_loss_batch + train_l2_loss_batch + train_option['vgg_lambda'] * perceptLossBatch
            train_rmse_batch = math.sqrt(train_l2_loss_batch.item() * 3)  # 3 channels rgb
            #backpropagation and update params
            optimizer.zero_grad()
            train_loss_batch.backward()
            optimizer.step()
            #record running time
            time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
            del polluted_train_batch, unpolluted_train_batch
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

def loadDataByPaths(datapath, idx=None,size=None,prj_fov_mask=None):
    if idx is not None:
        data = readImgsMT(datapath, size=size, index=idx)
    else:
        data = readImgsMT(datapath, size=size)
    if prj_fov_mask is not None:
        data = torch.where(prj_fov_mask, data, torch.tensor([0.]))
    return data

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
    elif 'vggLoss' in loss_option:
        pass
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

    if cam_im.shape[0] > 5:
        grid_w = 5
        idx = random.sample(range(0, cam_im.shape[0]), grid_w)
    else:
        grid_w = cam_im.shape[0]
        idx = random.sample(range(0, cam_im.shape[0]), grid_w)
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
    prj_pred = model(data['cam'])
    if type(prj_pred) == tuple and len(prj_pred) > 1: prj_pred = prj_pred[0]
    return prj_pred


# evaluate model on validation dataset
def evaluate(model, valid_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cam_valid = valid_data['cam_valid']
    prj_valid = valid_data['prj_valid']

    with torch.no_grad():
        model.eval()  # explicitly set to eval mode

        # if you have limited GPU memory, we need to predict in batch mode
        last_loc = 0
        valid_mse, valid_ssim = 0., 0.

        prj_valid_pred = torch.zeros(prj_valid.shape)
        num_valid = cam_valid.shape[0]
        batch_size = 50 if num_valid > 50 else num_valid  # default number of test images per dataset

        for i in range(0, num_valid // batch_size):
            idx = range(last_loc, last_loc + batch_size)
            cam_valid_batch = cam_valid[idx, :, :, :].to(device) if cam_valid.device.type != 'cuda' else cam_valid[idx, :, :, :]
            prj_valid_batch = prj_valid[idx, :, :, :].to(device) if prj_valid.device.type != 'cuda' else prj_valid[idx, :, :, :]
            # predict batch
            prj_valid_pred_batch = model(cam_valid_batch,cam_valid_batch).detach()
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

#generate training title string
def optionToString(train_option):
    return '{}_{}_{}_{}_{}_{:.2f}'.format(train_option['model_name'], train_option['loss'],
                                                  train_option['nums_train'], train_option['batch_size'], train_option['epoch']
                                                  ,train_option['weight'])

