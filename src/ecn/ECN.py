import torch
import torch.nn as nn

class ECN(nn.Module):
    """
    :brief :this is the network of ECN
    :return :the output of ECN ,which's shape is [batch_size,channels,width,height].
    """
    def __init__(self):
        super(ECN, self).__init__()
        self.name = 'ECN'
        self.relu = nn.ReLU()

        #the surface net
        self.conv1_s = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2_s = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3_s = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4_s = nn.Conv2d(128, 256, 3, 1, 1)

        #the main net
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)
        #the transpose net
        self.transConv1 = nn.ConvTranspose2d(128, 64, 2,  2, 0)
        self.transConv2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.conv6 = nn.Conv2d(32, 3, 3, 1, 1)

        #the skip convolution
        self.skipConv1 = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu,
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu,
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu
        )

        self.skipConv2 = nn.Conv2d(32, 64, 1, 1, 0)
        self.skipConv3 = nn.Conv2d(64, 128, 1, 1, 0)

        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    def forward(self, x, s):
        #the surface residental
        res1_s = self.relu( self.conv1_s(s))
        res2_s = self.relu( self.conv2_s(res1_s))
        res3_s = self.relu( self.conv3_s(res2_s))
        res4_s = self.relu( self.conv4_s(res3_s))

        res1 = self.skipConv1(x)
        x = self.relu(self.conv1(x)+res1_s)
        res2 = self.skipConv2(x)
        x = self.relu(self.conv2(x)+res2_s)
        res3 = self.skipConv3(x)
        x = self.relu(self.conv3(x)+res3_s)

        x = self.relu(self.conv4(x)+res4_s)
        x = self.relu(self.conv5(x)+res3)
        x = self.relu(self.transConv1(x) + res2)
        x = self.relu(self.transConv2(x))
        x = self.relu(self.conv6(x)+res1)
        x = torch.clamp(x, min=0, max=1)
        return x



