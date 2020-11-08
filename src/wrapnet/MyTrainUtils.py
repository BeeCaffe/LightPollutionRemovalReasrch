from src.wrapnet.utils import *
import torch.nn.functional as F
import torch.optim as optim
import time
import cv2
DEBUG = True

l1_fun = nn.L1Loss()
l2_fun = nn.MSELoss()
ssim_fun = pytorch_ssim.SSIM()

class TrainProcess():
    def __init__(self, model_pro, model_back, warpping_net_model, valid_data, cam_path_dict, train_option):
        self.l1_fun = nn.L1Loss()
        self.l2_fun = nn.MSELoss()
        self.ssim_fun = pytorch_ssim.SSIM()
        self.model_back = model_back
        self.model_pro = model_pro
        self.warpping_net_model = warpping_net_model
        self.valid_data = valid_data
        self.cam_path_dict = cam_path_dict
        self.train_option = train_option
        self.device = self.train_option.get('device')
        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(self.device)
        self.weight.data.fill_(1.)

    def Train(self):
        if self.device.type == 'cuda' : torch.cuda.empty_cache()
        params_back = filter(lambda param: param.requires_grad, self.model_back.parameters())
        params_pro = filter(lambda param: param.requires_grad, self.model_pro.parameters())

        optimizer_back= optim.Adam(params_back, lr=self.train_option.get('lr'), weight_decay=self.train_option.get('l2_reg'))
        optimizer_pro = optim.Adam(params_pro, lr=self.train_option.get('lr'), weight_decay=self.train_option.get('l2_reg'))

        lr_scheduler_back = optim.lr_scheduler.StepLR(optimizer_back, step_size=self.train_option.get('lr_drop_rate'),
                                                 gamma=self.train_option.get('lr_drop_ratio'))
        lr_scheduler_pro = optim.lr_scheduler.StepLR(optimizer_pro, step_size=self.train_option.get('lr_drop_rate'),
                                                 gamma=self.train_option.get('lr_drop_ratio'))
        start_time = time.time()
        cam_train_path = fullfile(self.train_option.get('dataset_root'), self.cam_path_dict['cam_train'])
        prj_train_path = fullfile(self.train_option.get('dataset_root'), self.cam_path_dict['prj_train'])
        valid_psnr, valid_rmse, valid_ssim = 0., 0., 0.
        for i in range(self.train_option['epoch']):
            iters = 0
            while iters < self.train_option.get('nums_train') // self.train_option['batch_size']:
                idx = [j for j in range(iters * self.train_option['batch_size'], (iters + 1) * self.train_option['batch_size'])]
                cam_train_batch = self.loadDataByPaths(cam_train_path, idx, size=self.train_option['train_size']).to(self.device)
                prj_train_batch = self.loadDataByPaths(prj_train_path, idx=idx, size=self.train_option['prj_size']).to(self.device)
                warpped_cam_train = self.warpping_net_model(cam_train_batch).detach()  # compensated prj input image x^{*}
                self.model_back.train()
                self.model_pro.train()
                back_train = torch.pow(warpped_cam_train, self.train_option['backspect_gamma'])
                pro_train = torch.pow(warpped_cam_train, self.train_option['prospect_gamma'])
                # if DEBUG:
                #     img = back_train[0].permute(1, 2, 0).mul(255).to('cpu').detach().numpy()
                #     img = np.uint8(img)
                #     cv2.imshow('back_train ', img)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                b = warpped_cam_train[:, 0, :, :]
                g = warpped_cam_train[:, 1, :, :]
                r = warpped_cam_train[:, 2, :, :]
                bc = torch.max(torch.max(b, g), r).unsqueeze(1)
                # if DEBUG:
                #     img = bc[0].permute(1, 2, 0).mul(255).to('cpu').detach().numpy()
                #     img = np.uint8(img)
                #     cv2.imshow('bc',img)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                bc_back = self.weight * bc
                bc_pro = torch.ones_like(bc) - self.weight * bc

                pred_back = self.model_back(back_train)
                pred_pro = torch.mul(pred_pro, bc_pro)
                pred_back = torch.mul(pred_back, bc_back)
                if DEBUG:
                    img = pred_back[0].permute(1, 2, 0).mul(255).to('cpu').detach().numpy()
                    img = np.uint8(img)
                    cv2.imshow('pred_back', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                if DEBUG:
                    img = pred_pro[0].permute(1, 2, 0).mul(255).to('cpu').detach().numpy()
                    img = np.uint8(img)
                    cv2.imshow('pred_pro', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                pred = torch.clamp(pred_back+pred_pro, max=1.)
                if DEBUG:
                    img = pred[0].permute(1, 2, 0).mul(255).to('cpu').detach().numpy()
                    img = np.uint8(img)
                    cv2.imshow('pred', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                train_loss_batch, train_l2_loss_batch = self.computeLoss(pred, prj_train_batch, self.train_option.get('loss'))
                # perceptual loss
                if 'vggLoss' in self.train_option['loss']:
                    perceptLossBatch = pLoss.perceptualLoss(pred, prj_train_batch)
                    train_loss_batch = train_loss_batch + train_l2_loss_batch + self.train_option[
                        'vgg_lambda'] * perceptLossBatch
                train_rmse_batch = math.sqrt(train_l2_loss_batch.item() * 3)  # 3 channels rgb
                # backpropagation and update params
                optimizer_back.zero_grad()
                optimizer_pro.zero_grad()
                train_loss_batch.backward()
                optimizer_back.step()
                optimizer_pro.step()

                # record running time
                time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
                del cam_train_batch, prj_train_batch, warpped_cam_train
                lr_scheduler_back.step()
                lr_scheduler_pro.step()
                iters += 1
                # plot train
                valid_psnr, valid_rmse, valid_ssim = 0., 0., 0.
                # validation
                if self.valid_data is not None:
                    valid_psnr, valid_rmse, valid_ssim, prj_valid_pred = self.evaluate(self.model_pro, self.model_back, valid_data=self.valid_data)
                print(
                    'Epoch:{:5d}|Iter:{:5d}|Time:{}|Train Loss:{:.4f}|Train RMSE: {:.4f}|Valid PSNR: {:7s}|Valid RMSE: {:6s}'
                    '|Valid SSIM: {:6s}'.format(i, iters, time_lapse, train_loss_batch.item(),
                                                                   train_rmse_batch,
                                                                   '{:>2.4f}'.format(valid_psnr) if valid_psnr else '',
                                                                   '{:.4f}'.format(valid_rmse) if valid_rmse else '',
                                                                   '{:.4f}'.format(valid_ssim) if valid_ssim else ''
                                                                   ))
        # Done training and save the last epoch model
        checkpoint_dir = 'checkpoint'
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        title = self.optionToString(self.train_option)
        back_file_name = fullfile(checkpoint_dir, title + 'BACK.pth')
        torch.save(self.model_back, back_file_name)
        pro_file_name = fullfile(checkpoint_dir, title + 'PRO.pth')
        torch.save(self.model_pro, pro_file_name)
        print('Checkpoint saved to {}\n'.format(back_file_name))
        return back_file_name, pro_file_name, valid_psnr, valid_rmse, valid_ssim
        pass

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

    def predict(self, model, data):
        prj_pred = model(data['cam'])
        if type(prj_pred) == tuple and len(prj_pred) > 1: prj_pred = prj_pred[0]
        return prj_pred

    # evaluate model on validation dataset
    def evaluate(self, model_pro, model_back, valid_data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cam_valid = valid_data['cam_valid']
        prj_valid = valid_data['prj_valid']

        with torch.no_grad():
            model_back.eval()  # explicitly set to eval mode
            model_pro.eval()  # explicitly set to eval mode
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
                back = model_back(cam_valid_batch).detach()
                pro = model_pro(cam_valid_batch).detach()
                prj_valid_pred_batch = torch.clamp(back + pro, max=1)
                if type(prj_valid_pred_batch) == tuple and len(prj_valid_pred_batch) > 1: prj_valid_pred_batch = \
                prj_valid_pred_batch[0]
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
    # generate training title string
    def optionToString(self, train_option):
        return '{}_{}_{}_{}_{}'.format(train_option['model_name'],
                                       train_option['loss'],
                                       train_option['nums_train'],
                                       train_option['batch_size'],
                                       train_option['epoch'],)

