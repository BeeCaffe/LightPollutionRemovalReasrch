import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np
import cv2
from src.tools.Utils import CombineImages1DXLim
DEBUG = True

def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)


class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                print(loss_name)
                self.losses[loss_name] = losses[loss_name].data
            else:
                self.losses[loss_name] += losses[loss_name].data

            if (i + 1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    loss_np = loss.cpu().numpy()
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]),
                                                                 Y=np.array([loss_np / self.batch]),
                                                                 opts={'xlabel': 'epochs', 'ylabel': loss_name,
                                                                       'title': loss_name})
                else:
                    loss_np = loss.cpu().numpy()
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss_np / self.batch]),
                                  win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def img_log(img, title=""):
    img = torch.clamp(img, max=1)
    img = img[0].permute(1, 2, 0).mul(255).to('cpu').detach().numpy()
    img = np.uint8(img)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img_save(img, dir=""):
    img = torch.clamp(img, max=1)
    img = img[0].permute(1, 2, 0).mul(255).to('cpu').detach().numpy()
    img = np.uint8(img)
    cv2.imwrite(dir, img)

def torch2img(tensor):
    img = tensor[0].permute(1, 2, 0).mul(255).to('cpu').detach().numpy()
    img = np.uint8(img)
    return img

def opt2Str(opt):
    return '{}_{}_{}_{}_{}'.format(opt.model_name,
                                opt.n_epochs,
                                opt.batchSize,
                                opt.pro_gamma,
                                opt.back_gamma)

def combine(pred, img, mask, weight=2., tag="notag", epoch=1, iter=1):
    pred_back, pred_pro = torch.chunk(pred, chunks=2, dim=1)
    mask = torch.cat([mask, mask, mask], dim=1)
    mask = guidedfilter(img, mask)
    mask_pro = torch.clamp(weight*mask, max=1, min=0)
    mask_back = torch.clamp(torch.ones_like(mask) - torch.clamp(weight*mask, max=1, min=0), max=1, min=0)

    back = torch.mul(pred_back, mask_back)
    pro = torch.mul(pred_pro, mask_pro)
    comb = torch.clamp(back + pro, max=1, min=0)
    p, c, w, h = comb.shape
    if iter%100 ==0 :
        img = CombineImages1DXLim([
            torch2img(img),
            torch2img(mask_pro),
            torch2img(mask_back),
            torch2img(pred_back),
            torch2img(pred_pro),
            torch2img(pro),
            torch2img(back),
            torch2img(comb)])
        cv2.imwrite("F:/yandl/temp/"+"epoch{:<d}_iter{:<d}_{:<s}_{:<s}".format(epoch, iter, "combine", tag) +'.jpg', img)
    return comb

def split(input_3c, img, mask, weight=2., gamma_pro=1, gamma_back=1, epoch=1, iter=1, tag="notag"):
    mask = torch.cat([mask, mask, mask], dim=1)
    mask = guidedfilter(img, mask)
    mask_pro = torch.clamp(weight * mask, max=1, min=0)
    mask_back = torch.clamp(torch.ones_like(mask) - torch.clamp(weight * mask, max=1, min=0), max=1, min=0)
    back = torch.mul(torch.pow(input_3c, gamma_back), mask_back)
    pro = torch.mul(torch.pow(input_3c, gamma_pro),  mask_pro)
    output_6c = torch.clamp(torch.cat([back, pro], dim=1), max=1, min=0)
    if iter%100 == 0:
        img = CombineImages1DXLim([
            torch2img(img),
            torch2img(mask_pro),
            torch2img(mask_back),
            torch2img(torch.pow(input_3c, gamma_pro)),
            torch2img(torch.pow(input_3c, gamma_back)),
            torch2img(pro),
            torch2img(back)])
        cv2.imwrite("F:/yandl/temp/"+"epoch{:<d}_iter{:<d}_{:<s}_{:<s}".format(epoch, iter, "split", tag) +'.jpg', img)
    return output_6c

def guidedfilter(img, mask, r=81, eps=0.001):
    Is = torch2cv(img)
    ps = torch2cv(mask)
    n = Is.shape[0]
    out_mask = ps
    for i in range(n):
        img = Is[i]
        mask = ps[i]
        height, width, c = img.shape
        m_I = cv2.boxFilter(img, -1, (r, r))
        m_p = cv2.boxFilter(mask, -1, (r, r))
        m_Ip = cv2.boxFilter(img * mask, -1, (r, r))
        cov_Ip = m_Ip - m_I * m_p
        m_II = cv2.boxFilter(img * img, -1, (r, r))
        var_I = m_II - m_I * m_I
        a = cov_Ip / (var_I + eps)
        b = m_p - a * m_I
        m_a = cv2.boxFilter(a, -1, (r, r))
        m_b = cv2.boxFilter(b, -1, (r, r))
        ps[i] = (m_a * img + m_b)
        out_mask[i] = cv2.cvtColor(cv2.cvtColor(ps[i], cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2torch(out_mask)

def gaussBlur(tensor):
    img = torch2cv(tensor)
    batch, _, _, _ = img.shape
    for i in range(batch):
        img[i] = cv2.GaussianBlur(img[i], ksize=(7, 7), sigmaX=4, sigmaY=4)
    return cv2torch(img)

def torch2cv(tensor):
    return tensor.permute(0, 2, 3, 1).mul(255).to('cpu').detach().numpy()

def cv2torch(imgs):
    imgs = torch.from_numpy(imgs)
    return imgs.permute((0, 3, 1, 2)).float().div(255).to('cuda')

def bright_channel(x):
    b = x[:, 0, :, :]
    g = x[:, 1, :, :]
    r = x[:, 2, :, :]
    bright_channel = torch.max(torch.max(b, g), r).unsqueeze(1)
    bright_channel = torch2cv(bright_channel)
    r = 7
    n, _,_,_ = bright_channel.shape
    for i in range(n):
        bright_channel[i,:,:,0] = cv2.erode(bright_channel[i,:,:,0], np.ones((2 * r + 1, 2 * r + 1)))
    return cv2torch(bright_channel)

def normalize(tensor):
    return tensor.permute((0, 3, 1, 2)).float().div(255)

