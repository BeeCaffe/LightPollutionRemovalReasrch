#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import numpy as np
import cv2
import torch

from src.cyclegan_pairnet.models import Generator
from src.cyclegan_pairnet.models import Discriminator
from src.cyclegan_pairnet.utils import ReplayBuffer
from src.cyclegan_pairnet.utils import LambdaLR
from src.cyclegan_pairnet.utils import Logger
from src.cyclegan_pairnet.utils import weights_init_normal
from src.cyclegan_pairnet.datasets import ImageDataset
from src.cyclegan_pairnet.utils import img_log
from src.cyclegan_pairnet.models import MaskNet
from src.cyclegan_pairnet.CorrectImage import compensate_img
from src.cyclegan_pairnet.utils import combine
from src.cyclegan_pairnet.utils import bright_channel
from src.cyclegan_pairnet.utils import opt2Str
from src.cyclegan_pairnet.utils import split
from src.cyclegan_pairnet.utils import cv2torch
from src.cyclegan_pairnet.utils import normalize
import os

DEBUG = False

"""
    A: camera captured image
    B: projector input image 
"""
parser = argparse.ArgumentParser()
torch.cuda.set_device(0)
parser.add_argument('--model_name', type=str, default='cyclegan-pairnet', help='root directory of the dataset')
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs of training')
parser.add_argument('--train_num', type=int, default=15000, help='how many image used to train')
parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/cyclecn-curve/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=1, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=6, help='number of channels of input data')
parser.add_argument('--output_nc',  type=int, default=6, help='number of channels of output data')
parser.add_argument('--cuda', action='store_false', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--pro_gamma', type=float, default=1., help='gamma correction value')
parser.add_argument('--back_gamma', type=float, default=1.2, help='gamma correction value')
opt = parser.parse_args()
print(opt)



if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.input_nc, opt.output_nc)
mask_net = MaskNet(3, out_channels=1)


netD_A = Discriminator(3)
netD_B = Discriminator(3)
masknet_params = filter(lambda param: param.requires_grad, mask_net.parameters())


if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)
mask_net.eval()

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
l2_loss = torch.nn.MSELoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_mask = torch.optim.Adam(masknet_params, lr=opt.lr, betas=(0.5, 0.999))



lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_MaskNet = torch.optim.lr_scheduler.LambdaLR(optimizer_mask,
                                                             lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)


# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, 3, opt.size, opt.size)
input_B = Tensor(opt.batchSize, 3, opt.size, opt.size)

target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader

dataloader = DataLoader(ImageDataset(opt.dataroot,
                                     unaligned=True,
                                     train_number=opt.train_num,),
                                     batch_size=opt.batchSize,
                                     shuffle=False,
                                     num_workers=opt.n_cpu,
                                     drop_last=True)
# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A_3c = Variable(input_A.copy_(normalize(batch['A'])))
        real_B_3c = Variable(input_B.copy_(normalize(batch['B'])))
        real_A_3c = torch.clamp(real_A_3c, max=1, min=0)
        real_B_3c = torch.clamp(real_B_3c, max=1, min=0)
        mask_A = mask_net(real_A_3c)
        mask_B = mask_net(real_B_3c)
        real_A_6c = split(real_A_3c, real_A_3c, mask_A, 2, gamma_back=1.2, gamma_pro=1.4)
        real_B_6c = split(real_B_3c, real_B_3c, mask_B, 2, gamma_back=1., gamma_pro=1.)

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()
        optimizer_mask.zero_grad()


        # # Identity loss
        # # G_A2B(B) should equal B if real B is fed
        # same_B_6c = netG_A2B(real_B_6c)
        # same_B_3c = combine(same_B_6c, real_B_3c, mask_A)
        # loss_identity_B = criterion_identity(same_B_3c, real_B_3c) * 10.0
        # # G_B2A(A) should equal A if real A is fed
        # same_A_6c = netG_B2A(real_A_6c)
        # same_A_3c = combine(same_A_6c, real_B_3c, mask_B)
        # loss_identity_A = criterion_identity(same_A_3c, real_A_3c) * 10.0


        # GAN loss
        # round 1
        fake_B_6c = netG_A2B(real_A_6c)
        fake_B_3c = combine(fake_B_6c, real_A_3c, mask_A, tag="gan_A2B")
        pred_fake = netD_B(fake_B_3c)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        # round2
        fake_A_6c = netG_B2A(real_B_6c)
        fake_A_3c = combine(fake_A_6c, real_B_3c, mask_B, tag="gan_B2A")
        pred_fake = netD_A(fake_A_3c)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        # round 1
        recovered_A_6c = netG_B2A(fake_B_6c)
        recovered_A_3c = combine(recovered_A_6c, real_A_3c, mask_A, tag="cycle_B2A")
        loss_cycle_ABA = criterion_cycle(recovered_A_3c, real_A_3c) * 10.0

        # round 2
        recovered_B_6c = netG_A2B(fake_A_6c)
        recovered_B_3c = combine(recovered_B_6c, real_B_3c, mask_B, tag="cycle_A2B")
        loss_cycle_BAB = criterion_cycle(recovered_B_3c, real_B_3c) * 10.0

        # Total loss
        # loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_Mask = l2_loss(mask_A, bright_channel(real_A_3c))

        loss_Mask.backward(retain_graph=True)
        loss_G.backward(retain_graph=True)

        optimizer_G.step()
        optimizer_mask.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A_3c)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A_3c = fake_A_buffer.push_and_pop(fake_A_3c)
        pred_fake = netD_A(fake_A_3c.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward(retain_graph=True)

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()
        optimizer_mask.zero_grad()

        # Real loss
        pred_real = netD_B(real_B_3c)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_3_6c = fake_B_buffer.push_and_pop(fake_B_3c)
        pred_fake = netD_B(fake_B_3c.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward(retain_graph=True)

        optimizer_D_B.step()
        optimizer_mask.step()

        ###################################

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_GAN_A2B': (loss_GAN_A2B),
                    'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB),
                    'loss_D': (loss_D_A + loss_D_B),
                    'loss_Mask':loss_Mask})
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    lr_scheduler_MaskNet.step()

    # Save models checkpoints
    save_checkpoint_path = 'checkpoint/'+opt.model_name+'/'+opt2Str(opt)+'/'
    if not os.path.exists(save_checkpoint_path):
        os.makedirs(save_checkpoint_path)
    torch.save(netG_A2B.state_dict(), save_checkpoint_path+'netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), save_checkpoint_path+'netG_B2A.pth')
    torch.save(netD_A.state_dict(), save_checkpoint_path+'netD_A.pth')
    torch.save(netD_B.state_dict(), save_checkpoint_path+'netD_B.pth')
    torch.save(mask_net, save_checkpoint_path+'masknet.pth')

del netG_A2B
del netG_B2A
del netD_A
del netD_B
del mask_net
torch.cuda.empty_cache()

with torch.no_grad():
    save_checkpoint_path = 'checkpoint/'+opt.model_name+'/'+opt2Str(opt)+'/'
    compensate_img(A2Bpth=save_checkpoint_path+'netG_A2B.pth',
                   B2Apth=save_checkpoint_path+'netG_B2A.pth',
                   Maskpth=save_checkpoint_path+'masknet.pth',
                   dataroot='input/cyclegancn/',
                   optstr=opt2Str(opt))
###################################
