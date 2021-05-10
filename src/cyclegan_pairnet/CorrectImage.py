#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import cv2

from src.cyclegan_pairnet .models import Generator
from src.cyclegan_pairnet.datasets import ImageDataset
from src.cyclegan_pairnet.utils import combine
from src.cyclegan_pairnet.utils import split
from src.cyclegan_pairnet.utils import img_log
from src.cyclegan_pairnet.utils import normalize
DEBUG = True

parser = argparse.ArgumentParser()
parser.add_argument('--generator_A2B', type=str, default='checkpoint/cyclegan-pairnet/cyclegan-pairnet_100_4_1.0_1.2/epoch_4/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='checkpoint/cyclegan-pairnet/cyclegan-pairnet_100_4_1.0_1.2/epoch_4/netG_B2A.pth', help='B2A generator checkpoint file')
parser.add_argument('--MaskNet', type=str, default='checkpoint/cyclegan-pairnet/cyclegan-pairnet_100_4_1.0_1.2/epoch_4/masknet.pth', help='B2A generator checkpoint file')

opt = parser.parse_args()

def compensate_img(A2Bpth, B2Apth, Maskpth, dataroot, save_filename, batchSize=1, size = (1920, 1080)):
    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(6, 6)
    netG_B2A = Generator(6, 6)

    netG_A2B.cuda()
    netG_B2A.cuda()

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(A2Bpth))
    netG_B2A.load_state_dict(torch.load(B2Apth))
    masknet = torch.load(Maskpth)


    # Set model's test.py mode
    netG_A2B.eval()
    netG_B2A.eval()
    masknet.eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor
    input_A = Tensor(batchSize, 3, size[1], size[0])
    input_B = Tensor(batchSize, 3, size[1], size[0])

    # Dataset loader

    dataloader = DataLoader(ImageDataset(dataroot, mode='test', img_size=size),
                            batch_size=batchSize, shuffle=False, num_workers=0)
    ###################################

    ###### Testing######

    # Create output dirs if they don't exist
    file_name = 'output/cyclegan-pairnet/' + save_filename
    if not os.path.exists(file_name+'/A'):
        os.makedirs(file_name+'/A')
    if not os.path.exists(file_name+'/B'):
        os.makedirs(file_name+'/B')
    if not os.path.exists(file_name + '/mask'):
        os.makedirs(file_name + '/mask')
    if not os.path.exists(file_name + '/b_mask'):
        os.makedirs(file_name + '/b_mask')

    for i, batch in enumerate(dataloader):
        # Set model input
        real_A_3c = Variable(input_A.copy_(normalize(batch['A'])))
        real_B_3c = Variable(input_B.copy_(normalize(batch['B'])))
        real_A_3c = torch.clamp(real_A_3c, max=1, min=0)
        real_B_3c = torch.clamp(real_B_3c, max=1, min=0)

        # masknet
        mask_A = masknet(real_A_3c)
        mask_B = masknet(real_B_3c)

        real_A_6c = split(real_A_3c, real_A_3c, mask_A, 2)
        real_B_6c = split(real_B_3c, real_B_3c, mask_B, 2)

        # Generate output
        fake_B_6c = netG_A2B(real_A_6c)
        fake_A_6c = netG_B2A(real_B_6c)
        fake_A_3c = combine(fake_A_6c, real_A_3c, mask_A)
        fake_B_3c = combine(fake_B_6c, real_B_3c, mask_A)

        # Save image files
        save_image(fake_A_3c, file_name+'/A/%04d.png' % (i+1))
        save_image(fake_B_3c, file_name+'/B/%04d.png' % (i+1))
        save_image(mask_A, file_name+'/mask/%04d.png' % (i+1))
        save_image(torch.clamp(torch.ones_like(mask_A) - torch.clamp(2.*mask_A, max=1, min=0), max=1, min=0),
                   file_name+'/b_mask/%04d.png' % (i+1))


        fake_A = cv2.resize(cv2.imread(file_name+'/A/%04d.png' % (i+1)), (1920, 1080))
        fake_B = cv2.resize(cv2.imread(file_name+'/B/%04d.png' % (i+1)), (1920, 1080))
        mask_A = cv2.resize(cv2.imread(file_name+'/mask/%04d.png' % (i+1)), (1920, 1080))
        mask_back = cv2.resize(cv2.imread(file_name+'/b_mask/%04d.png' % (i+1)), (1920, 1080))

        cv2.imwrite(file_name+'/A/%04d.png'%(i+1), fake_A)
        cv2.imwrite(file_name+'/B/%04d.png'%(i+1), fake_B)
        cv2.imwrite(file_name+'/mask/%04d.png'%(i+1), mask_A)
        cv2.imwrite(file_name+'/b_mask/%04d.png'%(i+1), mask_back)


        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))
        torch.cuda.empty_cache()

    sys.stdout.write('\n')
    ###################################

if __name__=="__main__":
    compensate_img(A2Bpth=opt.generator_A2B,
                   B2Apth=opt.generator_B2A,
                   Maskpth=opt.MaskNet,
                   dataroot='input/cyclegancn/',
                   save_filename="temp")