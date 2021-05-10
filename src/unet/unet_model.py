""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, residel_num=3):
        super(UNet, self).__init__()
        self.name = "BCCN_Sub_Net"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU()
        self.residel_num = residel_num
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
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.up3 = nn.ConvTranspose2d(32, 3, 2, 2, 0)
        self.up4 = nn.ConvTranspose2d(3, 3, 2, 2, 0)
        self.out_up = nn.Conv2d(32, 3, 3, 1, 1)
        self.res1 = residual_block(128)
        self.res2 = residual_block(self.out_channels)
        self.skipConv = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu,
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu,
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu
        )

    def forward(self, x):
        x_skip = self.skipConv(x)
        #prospect compensation net
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = x3
        if self.residel_num != 0:
            x = self.res1(x)
            for i in range(0, self.residel_num-1):
                x = self.res1(x)
        else:
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            pass

        x = self.relu(self.up1(x)+x2)       #(64,64,64)
        x = self.relu(self.up2(x)+x1)       #(32,128,128)
        x = self.relu(self.up3(x)+x_skip)   #(3,256,256)
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