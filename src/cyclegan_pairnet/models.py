import torch.nn as nn
import torch.nn.functional as F
import torch
from src.cyclegan_pairnet.utils import img_log

DEBUG = True

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorB2A(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(GeneratorB2A, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self,  inchannel=6, outchannels=6, res_block_num=9):
        super(Generator, self).__init__()
        self.model = ResSubNet(inchannel, outchannels, res_block_num = res_block_num)
    def forward(self, x):
        x = self.model(x)
        return x

class ResSubNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=6, res_block_num=9):
        super(ResSubNet, self).__init__()
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
        x = torch.clamp(x, max=1, min=0)
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

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

class MaskNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaskNet, self).__init__()
        self.name = "Reflection_Channel_Net"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ).to('cuda')
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ).to('cuda')
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2, 0).to('cuda')
        self.up3 = nn.ConvTranspose2d(32, out_channels, 2, 2, 0).to('cuda')
        self.out_up = nn.Conv2d(3, 1, 3, 1, 1)
        self.res1 = residual_block(64).to('cuda')
        self.res2 = residual_block(self.out_channels).to('cuda')

    def forward(self, x):
        # prospect compensation net
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x = self.res1(x2)
        x = self.relu(self.up2(x))  # (32,128,128)
        x = self.relu(self.up3(x))  # (3,256,256)
        x = torch.clamp(x, max=1)
        return x
