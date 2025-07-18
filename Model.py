import torch
import torch.nn as nn
from torch.fft import rfft, irfft, rfftn, irfftn
from torch.nn import functional as F
from einops.layers.torch import Rearrange

'''
RTUnet: 2D mapping along both theta and p axis
RTFnet: 1D fourier transform and filter along p axis & 2D conv along theta axis
'''

class RTpad(nn.Module):
    def __init__(self, pad_width, if_zero=False):
        super(RTpad, self).__init__()
        self.pad_width = pad_width
        self.if_zero = if_zero

    def forward(self, tensor):
        if self.if_zero:
            tensor = torch.nn.functional.pad(tensor, pad=(self.pad_width, self.pad_width, 0, 0),
                                             mode='constant', value=0)
        pad_left = torch.flip(tensor[:, :, -self.pad_width:, :], dims=[3])
        pad_right = torch.flip(tensor[:, :, 0:self.pad_width, :], dims=[3])
        tensor_pad1 = torch.cat([pad_left, tensor, pad_right], dim=2)
        return tensor_pad1


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad_width=1, RT=False):
        super(DoubleConv, self).__init__()

        if RT:
            self.conv1 = nn.Sequential(
                RTpad(pad_width=pad_width, if_zero=True),
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, stride=1)
            )
            self.conv2 = nn.Sequential(
                RTpad(pad_width=1, if_zero=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, stride=1)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad_width, stride=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x



class Filter_1D(nn.Module):
    def __init__(self, filter_h, filter_w, channel=1):
        super(Filter_1D, self).__init__()
        self.scale = 1/(channel)
        self.pad = filter_w//2
        self.complex_weight = nn.Parameter(torch.randn(channel, filter_h, filter_w + 1, 2, dtype=torch.float32)*self.scale)

    def forward(self, x):
        x = torch.nn.functional.pad(x, pad=(self.pad, self.pad, 0, 0),
                                             mode='constant', value=0)
        # print(x.shape)
        x = rfft(x, dim=-1, norm='ortho')
        # print(x.shape)
        weight = self.complex_weight
        assert weight.shape[1:3] == x.shape[2:4]
        weight = torch.view_as_complex(weight.contiguous())
        x = x * weight
        y = irfft(x, dim=-1, norm='ortho')
        return y[:, :, :, self.pad:3*self.pad]
class SF_1D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 filter_h, filter_w, RT_pad=False):
        super(SF_1D, self).__init__()
        self.iniF = nn.Conv2d(in_channels, out_channels//2, kernel_size=3,
                      stride=1, padding=1)
        self.F = Filter_1D(filter_h, filter_w, out_channels//2)
        self.S = DoubleConv(in_channels, out_channels//2, kernel_size=3, pad_width=1, RT=RT_pad)
    def forward(self, x):
        xf0 = self.iniF(x)
        xf = self.F(xf0)
        xs = self.S(x)
        x = torch.cat([xs, xf], dim=1)
        return x

class RTMnet(nn.Module):
    def __init__(self, RT_pad=False, dim=16, filter_size=256):
        super(RTMnet, self).__init__()
        self.conv0 = SF_1D(1, dim, filter_h=filter_size, filter_w=filter_size, RT_pad=RT_pad)
        self.pool1 = nn.MaxPool2d(2)
        self.conv1 = SF_1D(dim, dim*2, filter_h=filter_size//2, filter_w=filter_size//2, RT_pad=RT_pad)
        self.pool2 = nn.MaxPool2d(2)
        self.conv2 = SF_1D(dim*2, dim*4, filter_h=filter_size//4, filter_w=filter_size//4, RT_pad=RT_pad)
        self.pool3 = nn.MaxPool2d(2)
        self.conv3 = SF_1D(dim*4, dim*4, filter_h=filter_size//8, filter_w=filter_size//8, RT_pad=RT_pad)  # Bottleneck

        self.up3 = nn.ConvTranspose2d(dim*4, dim*4, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(dim*8, dim*2, kernel_size=3, pad_width=1, RT=RT_pad)
        self.up2 = nn.ConvTranspose2d(dim*2, dim*2, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(dim*4, dim, kernel_size=3, pad_width=1, RT=RT_pad)
        self.up1 = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(dim*2, dim//2, kernel_size=3, pad_width=1, RT=RT_pad)

        self.out = DoubleConv(dim//2, 1, kernel_size=3, pad_width=1, RT=RT_pad)

    def forward(self, x):
        x0 = self.conv0(x)  # [8,512,*]
        x1 = self.conv1(self.pool1(x0))  # [16,256,*]
        x2 = self.conv2(self.pool2(x1))  # [32,128,*]
        x3 = self.conv3(self.pool3(x2))  # [32,64,*]

        x = self.up3(x3)  # [32,128,*]
        x = torch.cat([x2, x], dim=1)  # [64,128,*]
        x = self.upconv3(x)  # [16,128,*]

        x = self.up2(x)  # [16,256,*]
        x = torch.cat([x1, x], dim=1)  # [32,256,*]
        x = self.upconv2(x)  # [8,256,*]

        x = self.up1(x)  # [8,512,*]
        x = torch.cat([x0, x], dim=1)  # [16,512,*]
        x = self.upconv1(x)  # [4,512,*]

        x = self.out(x)  # [1,512,*]
        return x




class TM(nn.Module):
    def __init__(self, input_size, output_size):
        super(TM, self).__init__()
        self.arrange1 = Rearrange('b c h w -> b c (h w)')
        self.linear = nn.Linear(input_size*input_size, output_size*output_size)
        self.arrange2 = Rearrange('b c (h w) -> b c h w', w=output_size)

    def forward(self, x):
        x = self.arrange1(x)
        x = self.linear(x)
        x = self.arrange2(x)
        return x

class Unet(nn.Module):
    def __init__(self, if_RT=True):
        super(Unet, self).__init__()
        self.conv0 = DoubleConv(1, 64, kernel_size=3, pad_width=1, RT=if_RT)
        self.pool1 = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(64, 128, RT=if_RT)
        self.pool2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(128, 256, RT=if_RT)
        self.pool3 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(256, 512, RT=if_RT)
        self.pool4 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(512, 512) #Bottleneck

        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.upconv4 = DoubleConv(1024, 256)
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(512, 128)
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(256, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(128, 32)

        self.out = DoubleConv(32, 1)

    def forward(self, x):
        x0 = self.conv0(x) #[64,256,*]
        x1 = self.conv1(self.pool1(x0)) #[128,128,*]
        x2 = self.conv2(self.pool2(x1)) #[256,64,*]
        x3 = self.conv3(self.pool3(x2)) #[512,32,*]
        x4 = self.conv4(self.pool4(x3)) #[512,16,*]

        x = self.up4(x4) #[512,32,*]
        x = torch.cat([x3, x], dim=1) #[1024,32,*]
        x = self.upconv4(x) #[256,32,*]

        x = self.up3(x) #[256,64,*]
        x = torch.cat([x2, x], dim=1) #[512,64,*]
        x = self.upconv3(x) #[128,64,*]

        x = self.up2(x) #[128,128,*]
        x = torch.cat([x1, x], dim=1) #[256,128,*]
        x = self.upconv2(x) #[64,128,*]

        x = self.up1(x) #[64,256,*]
        x = torch.cat([x0, x], dim=1) #[128,256,*]
        x = self.upconv1(x) #[32,256,*]

        x = self.out(x) #[1,256,*]
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
