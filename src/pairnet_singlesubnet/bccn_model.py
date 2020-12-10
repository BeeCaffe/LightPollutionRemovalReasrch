import torch
from torch import nn
import cv2
import numpy as np
DEBUG = True
class Bccn_Net(nn.Module):
    def __init__(self, weight, res_block_num=3):
        super(Bccn_Net, self).__init__()
        self.res_block = residual_block(3)
        self.bccn_model = Bccn_Sub_Net(6, 6, res_block_num = res_block_num)
        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight.data.fill_(weight)
        # increase the resolution of the output layer
        self.conv7 = nn.Conv2d(3, 3 * 2 ** 2, kernel_size=3,
                               padding=1)  # (in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x, mask):
        x = self.bccn_model(x)
        mask_pro = torch.clamp(self.weight * mask, max=1)
        mask_back = torch.ones_like(mask) - torch.clamp(self.weight * mask, max=1)
        pred_back, pred_pro = torch.chunk(x, chunks=2, dim=1)
        back = torch.mul(pred_back, mask_back)
        pro = torch.mul(pred_pro, mask_pro)
        x = torch.clamp(back+pro, max=1)
        return x

class Bccn_Sub_Net(nn.Module):
    def __init__(self, in_channels=6, out_channels=6, res_block_num=3):
        super(Bccn_Sub_Net, self).__init__()
        self.name = "BCCN_Sub_Net"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_block_num = res_block_num
        self.relu = nn.ReLU()
        self.down1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.up3 = nn.ConvTranspose2d(32, self.out_channels, 2, 2, 0)
        self.up4 = nn.ConvTranspose2d(self.out_channels, 3, 2, 2, 0)
        self.out_up = nn.Conv2d(32, 3, 3, 1, 1)
        self.res1 = residual_block(128)
        self.res2 = residual_block(self.out_channels)
        self.skipConv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1),
            self.relu,
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1),
            self.relu,
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1),
            self.relu
        )

    def forward(self, x):
        x_in = self.skipConv(x)
        #prospect compensation net
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.res1(x3)
        x = self.res1(x)
        x = self.res1(x)
        if self.res_block_num > 3:
            for i in range(self.res_block_num-3):
                x = self.res1(x)
        x = self.relu(self.up1(x)+x2)#(64,64,64)
        x = self.relu(self.up2(x)+x1)#(32,128,128)
        x = self.relu(self.up3(x)+x_in)#(3,256,256)
        x = torch.clamp(x, max=1)
        return x

class residual_block(nn.Module):
    def __init__(self, channels):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual

class MaskNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaskNet, self).__init__()
        self.name = "Reflection_Channel_Net"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU()
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.up3 = nn.ConvTranspose2d(32, 3, 2, 2, 0)
        self.up4 = nn.ConvTranspose2d(3, 3, 2, 2, 0)
        self.out_up = nn.Conv2d(3, 1, 3, 1, 1)
        self.res1 = residual_block(64)
        self.res2 = residual_block(self.out_channels)

    def forward(self, x):
        # prospect compensation net
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x = self.res1(x2)
        x = self.relu(self.up2(x))  # (32,128,128)
        x = self.relu(self.up3(x))  # (3,256,256)
        x = torch.clamp(x, max=1)
        return x

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
