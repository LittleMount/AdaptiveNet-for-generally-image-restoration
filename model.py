# 创建网络模型
# 内容：double、
import torch
import torch.nn as nn
import torch.nn.functional as F

#************ UNet模型 ************#
class DoubleConv(nn.Module):
    """定义UNET网络中的卷积块，由两个卷积层组成"""

    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.up_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
    def forward(self, x, t):
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return self.double_conv(x)+emb

class Ot_embedding(nn.Module):
    def __init__(self, in_channels=1, dim=8):
        super(Ot_embedding, self).__init__()
        self.ot_embedding = nn.Sequential(
            # 448*448
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True),
            # 224*224
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim * 2),
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim * 4),
            nn.ReLU(inplace=True),
            # 112*112
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim*4),
            nn.Conv2d(dim * 4, dim * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim * 8),
            nn.ReLU(inplace=True),
            # 56*56
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(dim * 8, dim * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim * 8),
            nn.Conv2d(dim * 8, dim * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim * 16),
            nn.ReLU(inplace=True),
            # 28*28
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(dim * 16, dim * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim * 16),
            nn.Conv2d(dim * 16, dim * 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim * 32),
            nn.ReLU(inplace=True),
            # 14*14
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(dim * 32, dim * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim * 16),
            nn.Conv2d(dim * 16, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(196, 256),
        )

    def forward(self, x):
        return self.ot_embedding(x)

class AdaptiveNet(nn.Module):
    """定义UNET网络的架构"""

    def __init__(self, in_channels=1, out_channels=1, dim=16):
        super().__init__()
        self.ot_embedding = Ot_embedding()
        self.down_conv1 = DoubleConv(in_channels, dim)
        self.down_conv2 = DoubleConv(dim, dim * 2)
        self.down_conv3 = DoubleConv(dim * 2, dim * 4)
        self.down_conv4 = DoubleConv(dim * 4, dim * 8)
        self.down_conv5 = DoubleConv(dim * 8, dim * 16)
        self.up_transpose1 = nn.ConvTranspose2d(dim * 16, dim * 8, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(dim * 16, dim * 8)
        self.up_transpose2 = nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(dim * 8, dim * 4)
        self.up_transpose3 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(dim * 4, dim * 2)
        self.up_transpose4 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(dim * 2, dim)
        self.out_conv = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)



    def forward(self, x):
        # 编码器部分
        t = self.ot_embedding(x)
        x1 = self.down_conv1(x, t)
        x2 = self.down_conv2(F.max_pool2d(x1, kernel_size=2, stride=2), t)
        x3 = self.down_conv3(F.max_pool2d(x2, kernel_size=2, stride=2), t)
        x4 = self.down_conv4(F.max_pool2d(x3, kernel_size=2, stride=2), t)
        # x5 = self.down_conv5(F.max_pool2d(x4, kernel_size=2, stride=2), t)
        # 解码器部分
        # x = self.up_transpose1(x5)
        # x = self.up_conv1(torch.cat([x, x4], dim=1), t)
        # x = self.up_transpose2(x)
        x = self.up_transpose2(x4)
        x = self.up_conv2(torch.cat([x, x3], dim=1), t)
        x = self.up_transpose3(x)
        x = self.up_conv3(torch.cat([x, x2], dim=1), t)
        x = self.up_transpose4(x)
        x = self.up_conv4(torch.cat([x, x1], dim=1), t)
        # 输出层
        x = self.out_conv(x)
        return x



