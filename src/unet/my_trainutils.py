from src.unet.utils import *
import torch.nn.functional as F
import torch.optim as optim
import time
import torch
import cv2
DEBUG = True
import src.unet.utils as utils
import src.unet.hsvloss as HSV

l1_fun = nn.L1Loss()
l2_fun = nn.MSELoss()
ssim_fun = pytorch_ssim.SSIM()

class TrainProcess():
    def __init__(self, model, warpping_net_model, valid_data, cam_path_dict, train_option):
        self.l1_fun = nn.L1Loss()
        self.l2_fun = nn.MSELoss()
        self.ssim_fun = pytorch_ssim.SSIM()
        self.model = model
        self.valid_data = valid_data
        self.warpping_net_model = warpping_net_model
        self.cam_path_dict = cam_path_dict
        self.train_option = train_option
        self.device = self.train_option.get('device')
        self.gamma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(self.device)
        self.gamma.data.fill_(1/train_option['gamma'])


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
                warpped_cam_train = torch.pow(warpped_cam_train, self.gamma)
                # # if DEBUG:
                # #     img = back_train[0].permute(1, 2, 0).mul(255).to('cpu').detach().numpy()
                # #     img = np.uint8(img)
                # #     cv2.imshow('back_train ', img)
                # #     cv2.waitKey(0)
                # #     cv2.destroyAllWindows()
                predict = self.model(warpped_cam_train)
                train_loss_batch, train_l2_loss_batch = self.computeLoss(predict, prj_train_batch, self.train_option.get('loss'))
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
                    valid_psnr, valid_rmse, valid_ssim, prj_valid_pred = self.evaluate(self.model,valid_data=self.valid_data)
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
        torch.save(self.model, model_file_name)
        print('Checkpoint saved to {}\n'.format(model_file_name))
        return model_file_name, valid_psnr, valid_rmse, valid_ssim, train_msg

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

        if 'hsvloss' in loss_option:
            hsvloss = HSV.hsvloss(prj_pred, prj_train)
            return 1000*train_loss+hsvloss, l2_loss

        if 'labloss' in loss_option:
            labloss = HSV.labloss(prj_pred, prj_train)
            return 1000*train_loss+labloss, l2_loss

        if 'klloss' in loss_option:
            klloss = HSV.klloss(prj_pred, prj_train)
            return train_loss+klloss, l2_loss

        return train_loss, l2_loss

    def predict(self, model, data):
        prj_pred = model(data['cam'])
        if type(prj_pred) == tuple and len(prj_pred) > 1: prj_pred = prj_pred[0]
        return prj_pred

    # evaluate model on validation dataset
    def evaluate(self, model, valid_data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cam_valid = valid_data['cam_valid']
        prj_valid = valid_data['prj_valid']

        with torch.no_grad():
            model.eval()  # explicitly set to eval mode
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
                predict = model(x=cam_valid_batch).detach()
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
        return '{}_{}_{}_{}_{}_{}_{}'.format(train_option['model_name'],
                                                train_option['loss'],
                                                train_option['nums_train'],
                                                train_option['batch_size'],
                                                train_option['epoch'],
                                                train_option['res_block_num'],
                                                train_option['gamma'])

