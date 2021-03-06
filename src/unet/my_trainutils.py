from src.unet.utils import *
import torch.nn.functional as F
import torch.optim as optim
import time
import cv2
DEBUG = True
import src.unet.utils as utils

l1_fun = nn.L1Loss()
l2_fun = nn.MSELoss()
ssim_fun = pytorch_ssim.SSIM()

class TrainProcess():
    def __init__(self, model, mask_net, warpping_net_model, valid_data, cam_path_dict, train_option):
        self.l1_fun = nn.L1Loss()
        self.l2_fun = nn.MSELoss()
        self.ssim_fun = pytorch_ssim.SSIM()
        self.model = model
        self.warpping_net_model = warpping_net_model
        self.valid_data = valid_data
        self.cam_path_dict = cam_path_dict
        self.train_option = train_option
        self.device = self.train_option.get('device')
        self.pro_gamma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(self.device)
        self.pro_gamma.data.fill_(1/train_option['pro_gamma'])
        self.back_gamma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(self.device)
        self.back_gamma.data.fill_(1/train_option['back_gamma'])
        self.mask_net = mask_net

    def Train(self):
        if self.device.type == 'cuda' : torch.cuda.empty_cache()
        params= filter(lambda param: param.requires_grad, self.model.parameters())
        optimizer = optim.Adam(params, lr=self.train_option.get('lr'), weight_decay=self.train_option.get('l2_reg'))
        lr_scheduler_back = optim.lr_scheduler.StepLR(optimizer,
                                                      step_size=self.train_option.get('lr_drop_rate'),
                                                      gamma=self.train_option.get('lr_drop_ratio'))
        start_time = time.time()
        cam_train_path = fullfile(self.train_option.get('dataset_root'), self.cam_path_dict['cam_train'])
        prj_train_path = fullfile(self.train_option.get('dataset_root'), self.cam_path_dict['prj_train'])
        valid_psnr, valid_rmse, valid_ssim = 0., 0., 0.
        train_msg = dict(
            loss=[],
            rmse=[],
            valid_psnr=[],
            valid_rmse=[],
            valid_ssim=[]
        )
        for i in range(self.train_option['epoch']):
            iters = 0
            while iters < self.train_option.get('nums_train') // self.train_option['batch_size']:
                idx = [j for j in range(iters * self.train_option['batch_size'], (iters + 1) * self.train_option['batch_size'])]
                cam_train_batch = self.loadDataByPaths(cam_train_path, idx, size=self.train_option['train_size']).to(self.device)
                prj_train_batch = self.loadDataByPaths(prj_train_path, idx=idx, size=self.train_option['prj_size']).to(self.device)
                warpped_cam_train = self.warpping_net_model(cam_train_batch).detach()  # compensated prj input image x^{*}
                self.model.train()
                back_train = torch.pow(warpped_cam_train, self.back_gamma)
                pro_train = torch.pow(warpped_cam_train, self.pro_gamma)
                # # if DEBUG:
                # #     img = back_train[0].permute(1, 2, 0).mul(255).to('cpu').detach().numpy()
                # #     img = np.uint8(img)
                # #     cv2.imshow('back_train ', img)
                # #     cv2.waitKey(0)
                # #     cv2.destroyAllWindows()
                b = warpped_cam_train[:, 0, :, :]
                g = warpped_cam_train[:, 1, :, :]
                r = warpped_cam_train[:, 2, :, :]
                cam_mask = torch.max(torch.max(b, g), r).unsqueeze(1)
                predict_mask = self.mask_net(warpped_cam_train)
                predict = self.model(warpped_cam_train, pro_train, back_train, predict_mask)
                train_loss_batch, train_l2_loss_batch = self.computeLoss(predict, prj_train_batch, self.train_option.get('loss'))
                if self.train_option['is_train_mask']:
                    mask_loss_batch = l2_fun(predict_mask, cam_mask)
                    train_loss_batch = train_loss_batch+mask_loss_batch
                # perceptual loss
                train_rmse_batch = math.sqrt(train_l2_loss_batch.item() * 3) # 3 channels rgb
                # backpropagation and update params
                optimizer.zero_grad()
                train_loss_batch.backward()
                optimizer.step()
                time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
                del cam_train_batch, prj_train_batch, warpped_cam_train
                lr_scheduler_back.step()
                iters += 1
                # plot train
                valid_psnr, valid_rmse, valid_ssim = 0., 0., 0.
                # validation
                if self.valid_data is not None:
                    valid_psnr, valid_rmse, valid_ssim, prj_valid_pred = self.evaluate(self.model, self.mask_net,valid_data=self.valid_data)
                print('Epoch:{:5d}|Iter:{:5d}|Time:{}|Train Loss:{:.4f}|Train RMSE: {:.4f}|Valid PSNR: {:7s}|Valid RMSE: {:6s}'
                    '|Valid SSIM: {:6s}'.format(i,
                                                iters,
                                                time_lapse,
                                                train_loss_batch.item(),
                                                train_rmse_batch,
                                                '{:>2.4f}'.format(valid_psnr) if valid_psnr else '',
                                                '{:.4f}'.format(valid_rmse) if valid_rmse else '',
                                                '{:.4f}'.format(valid_ssim) if valid_ssim else ''))
                train_msg['loss'].append(train_loss_batch.item())
                train_msg['rmse'].append(train_rmse_batch)
                train_msg['valid_psnr'].append(valid_psnr)
                train_msg['valid_rmse'].append(valid_rmse)
                train_msg['valid_ssim'].append(valid_ssim)
        # Done training and save the last epoch model
        checkpoint_dir = "checkpoint/"+self.train_option['model_name']+'/'
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        title = self.optionToString(self.train_option)
        model_file_name = fullfile(checkpoint_dir, title + '.pth')
        mask_model_name = fullfile(checkpoint_dir, title + '-mask.pth')
        torch.save(self.model, model_file_name)
        torch.save(self.mask_net, mask_model_name)
        print('Checkpoint saved to {}\n'.format(model_file_name))
        return model_file_name, mask_model_name, valid_psnr, valid_rmse, valid_ssim, train_msg

    def loadDataByPaths(self, datapath, idx=None, size=None, prj_fov_mask=None):
        if idx is not None:
            data = readImgsMT(datapath, size=size, index=idx)
        else:
            data = readImgsMT(datapath, size=size)
        if prj_fov_mask is not None:
            data = torch.where(prj_fov_mask, data, torch.tensor([0.]))
        return data

    def computeLoss(self, prj_pred, prj_train, loss_option):
        # l1
        l1_loss = 0
        if 'l1' in loss_option:
            l1_loss = l1_fun(prj_pred, prj_train)
        # l2
        l2_loss = l2_fun(prj_pred, prj_train)
        # ssim
        ssim_loss = 0
        if 'ssim' in loss_option:
            ssim_loss = 1 * (1 - self.ssim_fun(prj_pred, prj_train))
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

    def predict(self, model, data):
        prj_pred = model(data['cam'])
        if type(prj_pred) == tuple and len(prj_pred) > 1: prj_pred = prj_pred[0]
        return prj_pred

    # evaluate model on validation dataset
    def evaluate(self, model, mask_net, valid_data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cam_valid = valid_data['cam_valid']
        prj_valid = valid_data['prj_valid']

        with torch.no_grad():
            model.eval()  # explicitly set to eval mode
            mask_net.eval()
            # explicitly set to eval mode
            # if you have limited GPU memory, we need to predict in batch mode
            last_loc = 0
            valid_mse, valid_ssim = 0., 0.
            prj_valid_pred = torch.zeros(prj_valid.shape)
            num_valid = cam_valid.shape[0]
            batch_size = 50 if num_valid > 50 else num_valid  # default number of test images per dataset

            for i in range(0, num_valid // batch_size):
                idx = range(last_loc, last_loc + batch_size)
                cam_valid_batch = cam_valid[idx, :, :, :].to(device) if cam_valid.device.type != 'cuda' else cam_valid[
                                                                                                             idx, :, :,
                                                                                                             :]
                prj_valid_batch = prj_valid[idx, :, :, :].to(device) if prj_valid.device.type != 'cuda' else prj_valid[
                                                                                                             idx, :, :,
                                                                                                             :]
                # predict batch
                predict_mask = mask_net(cam_valid_batch)
                back_test = torch.pow(cam_valid_batch, self.back_gamma)
                pro_test = torch.pow(cam_valid_batch, self.pro_gamma)
                predict = model(x=cam_valid_batch, x_pro=pro_test, x_back=back_test, mask=predict_mask).detach()
                prj_valid_pred_batch = torch.clamp(predict, max=1)
                if type(prj_valid_pred_batch) == tuple and len(prj_valid_pred_batch) > 1: prj_valid_pred_batch = \
                prj_valid_pred_batch[0]
                prj_valid_pred[last_loc:last_loc + batch_size, :, :, :] = prj_valid_pred_batch.cpu()

                # compute loss
                valid_mse += l2_fun(prj_valid_pred_batch, prj_valid_batch).item() * batch_size
                valid_ssim += ssim(prj_valid_pred_batch, prj_valid_batch) * batch_size
                # if DEBUG:
                #     img = prj_valid_pred_batch[0].permute(1, 2, 0).mul(255).to('cpu').detach().numpy()
                #     img = np.uint8(img)
                #     cv2.imshow('img'
                #                ' ', img)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                last_loc += batch_size
            # average
            valid_mse /= num_valid
            valid_ssim /= num_valid

            valid_rmse = math.sqrt(valid_mse * 3)  # 3 channel image
            valid_psnr = 10 * math.log10(1 / valid_mse)

        return valid_psnr, valid_rmse, valid_ssim, prj_valid_pred
    # generate training title string
    def optionToString(self, train_option):
        return '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(train_option['model_name'],
                                                train_option['loss'],
                                                train_option['nums_train'],
                                                train_option['batch_size'],
                                                train_option['epoch'],
                                                train_option['res_block_num'],
                                                train_option['back_gamma'],
                                                train_option['pro_gamma'],
                                                train_option['is_train_mask'])

