from src.unet.utils import *
import torch.nn.functional as F
import torch.optim as optim
import time
import torch
import cv2
DEBUG = False
from src.cyclegan_pairnet.utils import CombineImages1DXLim
from src.cyclegan_pairnet.utils import torch2img



l1_fun = nn.L1Loss()
l2_fun = nn.MSELoss()
ssim_fun = pytorch_ssim.SSIM()

class TrainProcess():
    def __init__(self, reflection_model, compensation_model, warpping_net_model, valid_data, cam_path_dict, train_option, refloss_weight):
        self.l1_fun = nn.L1Loss()
        self.l2_fun = nn.MSELoss()
        self.ssim_fun = pytorch_ssim.SSIM()
        self.reflection_model = reflection_model
        self.compensation_model = compensation_model
        self.valid_data = valid_data
        self.warpping_net_model = warpping_net_model
        self.cam_path_dict = cam_path_dict
        self.train_option = train_option
        self.device = self.train_option.get('device')
        self.gamma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(self.device)
        self.gamma.data.fill_(1/train_option['gamma'])
        self.refloss_weight = refloss_weight


    def Train(self):
        if self.device.type == 'cuda' : torch.cuda.empty_cache()
        params_reflection = filter(lambda param: param.requires_grad, self.reflection_model.parameters())
        params_compensation = filter(lambda param: param.requires_grad, self.compensation_model.parameters())

        optimizer_reflection = optim.Adam(params_reflection, lr=self.train_option.get('lr'), weight_decay=self.train_option.get('l2_reg'))
        optimizer_compensation = optim.Adam(params_compensation, lr=self.train_option.get('lr'), weight_decay=self.train_option.get('l2_reg'))

        lr_scheduler_back_reflection = optim.lr_scheduler.StepLR(optimizer_reflection,
                                                      step_size=self.train_option.get('lr_drop_rate'),
                                                      gamma=self.train_option.get('lr_drop_ratio'))
        lr_scheduler_back_compensation = optim.lr_scheduler.StepLR(optimizer_compensation,
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
                self.reflection_model.train()
                self.compensation_model.train()
                warpped_cam_train = torch.pow(warpped_cam_train, self.gamma)
                reflection_light_predict = self.reflection_model(prj_train_batch)
                compensated_predict = self.compensation_model(prj_train_batch)

                combine = torch.clamp(reflection_light_predict+compensated_predict, min=0, max=1)

                real_refelction_batch = torch.clamp(warpped_cam_train-prj_train_batch, min=0, max=1)
                reflection_light_loss = l1_fun(reflection_light_predict, real_refelction_batch) + \
                                        l2_fun(reflection_light_predict, real_refelction_batch)
                reflection_light_loss = (1/self.refloss_weight)*reflection_light_loss


                train_loss_batch, train_l2_loss_batch = self.computeLoss(combine, prj_train_batch, self.train_option.get('loss'))
                train_loss_batch = train_loss_batch
                train_rmse_batch = math.sqrt(train_l2_loss_batch.item() * 3) # 3 channels rgb
                optimizer_compensation.zero_grad()
                optimizer_reflection.zero_grad()

                train_loss_batch.backward(retain_graph=True)
                reflection_light_loss.backward(retain_graph=True)

                if iters % 100 == 0:
                    img = CombineImages1DXLim([
                        torch2img(prj_train_batch),
                        torch2img(warpped_cam_train),
                        torch2img(real_refelction_batch),
                        torch2img(reflection_light_predict),
                        torch2img(compensated_predict),
                        torch2img(combine)
                    ])
                    cv2.imwrite("F:/yandl/substractnet/temp/" + "epoch{:<d}_iter{:<d}_{:<s}_{:<s}".format(i, iters, "train",
                                                                                             "notag") + '.jpg', img)

                optimizer_reflection.step()
                optimizer_compensation.step()
                time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
                del cam_train_batch, prj_train_batch, warpped_cam_train
                lr_scheduler_back_compensation.step()
                lr_scheduler_back_reflection.step()
                iters += 1
                # plot train
                valid_psnr, valid_rmse, valid_ssim = 0., 0., 0.
                # validation
                if self.valid_data is not None:
                    valid_psnr, valid_rmse, valid_ssim, prj_valid_pred = self.evaluate(self.reflection_model,
                                                                                       self.compensation_model,
                                                                                       valid_data=self.valid_data)
                print('Epoch:{:5d}|Iter:{:5d}|Time:{}|ReflectionNet Loss:{:.4f}|CompensationNet Loss:{:.4f}|Train RMSE: {:.4f}|Valid PSNR: {:7s}|Valid RMSE: {:6s}'
                    '|Valid SSIM: {:6s}'.format(i,
                                                iters,
                                                time_lapse,
                                                reflection_light_loss.item(),
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
        checkpoint_dir = "checkpoint/"+'compensationnet'+'/'
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        title_reflection = 'reflect_' + self.optionToString(self.train_option)
        title_compensation = 'compensate_' + self.optionToString(self.train_option)

        reflection_model_name = fullfile(checkpoint_dir,title_reflection + '.pth')
        compensation_model_name = fullfile(checkpoint_dir, title_compensation + '.pth')

        torch.save(self.reflection_model, reflection_model_name)
        torch.save(self.compensation_model, compensation_model_name)

        print('Checkpoint saved to {}\n'.format(reflection_model_name))
        return reflection_model_name, compensation_model_name, valid_psnr, valid_rmse, valid_ssim, train_msg

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
        return train_loss, l2_loss

    def predict(self, model, data):
        prj_pred = model(data['cam'])
        if type(prj_pred) == tuple and len(prj_pred) > 1: prj_pred = prj_pred[0]
        return prj_pred

    # evaluate model on validation dataset
    def evaluate(self, reflection_model, compensation_model, valid_data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cam_valid = valid_data['cam_valid']
        prj_valid = valid_data['prj_valid']

        with torch.no_grad():
            reflection_model.eval()  # explicitly set to eval mode
            compensation_model.eval()  # explicitly set to eval mode

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
                reflection_predict = reflection_model(x=cam_valid_batch).detach()
                compensation_predict = compensation_model(x=prj_valid_batch).detach()

                prj_valid_pred_batch = torch.clamp(reflection_predict+compensation_predict, min=0, max=1)

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
        return '{}_{}_{}_{}_{}_{}_{}_{}'.format(train_option['model_name'],
                                                train_option['loss'],
                                                train_option['nums_train'],
                                                train_option['batch_size'],
                                                train_option['epoch'],
                                                train_option['res_block_num'],
                                                train_option['gamma'],
                                                int(1./train_option['refloss_weight']))

