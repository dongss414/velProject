import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class Res_Bottle_Block_down(nn.Module):
    def __init__(self, input, output, kernel_size, stride, padding, bias=True):
        super(Res_Bottle_Block_down, self).__init__()
        middle = output // 4
        self.conv1 = nn.Conv3d(input, middle, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv2 = nn.Conv3d(middle, middle, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv3 = nn.Conv3d(middle, output, kernel_size=1, stride=1, padding=0, bias=bias)
        self.downsample = nn.Conv3d(input, output, kernel_size=stride, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        xx = self.downsample(x)
        x = self.conv1(F.leaky_relu(x, negative_slope=0.2))
        x = self.conv2(F.leaky_relu(x, negative_slope=0.2))
        x = self.conv3(F.leaky_relu(x, negative_slope=0.2))
        x = x + xx
        return x


class Res_Block_down(nn.Module):
    def __init__(self, input, output, kernel_size, stride, padding, bias=True):
        super(Res_Block_down, self).__init__()
        self.conv1 = nn.Conv3d(input, output, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv2 = nn.Conv3d(output, output, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)
        self.downsample = nn.Conv3d(input, output, kernel_size=stride, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        xx = self.downsample(x)
        x = self.conv1(F.leaky_relu(x, negative_slope=0.2))
        x = self.conv2(F.leaky_relu(x, negative_slope=0.2))
        x = x + xx
        return x


class Res_Bottle_Block_up(nn.Module):
    def __init__(self, input, output, kernel_size, stride, padding, output_padding=1, bias=True):
        super(Res_Bottle_Block_up, self).__init__()
        middle = output // 4
        self.conv1 = nn.Conv3d(input, middle, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv2 = nn.ConvTranspose3d(middle, middle, kernel_size=kernel_size, stride=stride, padding=padding,
                                        output_padding=output_padding, bias=bias)
        self.conv3 = nn.Conv3d(middle, output, kernel_size=1, stride=1, padding=0, bias=bias)
        self.upsample = nn.ConvTranspose3d(input, output, kernel_size=stride, stride=stride, padding=0,
                                           output_padding=0, bias=bias)

    def forward(self, x):
        xx = self.upsample(x)
        x = self.conv1(F.leaky_relu(x, negative_slope=0.2))
        x = self.conv2(F.leaky_relu(x, negative_slope=0.2))
        x = self.conv3(F.leaky_relu(x, negative_slope=0.2))
        x = x + xx
        return x


class Res_Block_up(nn.Module):
    def __init__(self, input, output, kernel_size, stride, padding, output_padding=1, bias=True):
        super(Res_Block_up, self).__init__()
        self.conv1 = nn.ConvTranspose3d(input, output, kernel_size=kernel_size, stride=stride, padding=padding,
                                        output_padding=output_padding, bias=bias)
        self.conv2 = nn.Conv3d(output, output, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)
        self.upsample = nn.ConvTranspose3d(input, output, kernel_size=stride, stride=stride, padding=0,
                                           output_padding=0, bias=bias)

    def forward(self, x):
        xx = self.upsample(x)
        x = self.conv1(F.leaky_relu(x, negative_slope=0.2))
        x = self.conv2(F.leaky_relu(x, negative_slope=0.2))
        x = x + xx
        return x


class Res_Bottle_Block(nn.Module):
    def __init__(self, input, output, kernel_size, stride, padding, bias=True, dropout_prob=0):
        super(Res_Bottle_Block, self).__init__()
        middle = output // 4
        self.conv1 = nn.Conv3d(input, middle, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv2 = nn.Conv3d(middle, middle, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)
        self.conv3 = nn.Conv3d(middle, output, kernel_size=1, stride=1, padding=0, bias=bias)
        self.sample = nn.Conv3d(input, output, kernel_size=1, stride=1, padding=0, bias=bias)
        self.dropout = nn.Dropout3d(p=dropout_prob)

    def forward(self, x):
        xx = self.sample(x)
        x = self.conv1(F.leaky_relu(x, negative_slope=0.2))
        x = self.dropout(x)
        x = self.conv2(F.leaky_relu(x, negative_slope=0.2))
        x = self.dropout(x)
        x = self.conv3(F.leaky_relu(x, negative_slope=0.2))
        x = x + xx
        return x


class Res_Block(nn.Module):
    def __init__(self, input, output, kernel_size, stride, padding, bias=True):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv3d(input, output, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv2 = nn.Conv3d(output, output, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.sample = nn.Conv3d(input, output, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        xx = self.sample(x)
        x = self.conv1(F.leaky_relu(x, negative_slope=0.2))
        x = self.conv2(F.leaky_relu(x, negative_slope=0.2))
        x = x + xx
        return x
class UNet3D_trainVelocity(nn.Module):
    # Network Architecture is exactly same as UNet+++ and 3D
    def __init__(self,inchannels=2,f=[8, 16, 32, 64, 128]):
        super(UNet3D_trainVelocity, self).__init__()
        # upsample and downsample
        # self.up2D_3D = nn.Upsample(scale_factor=(64,1,1), mode='trilinear')
        self.up2 = nn.Upsample(scale_factor=(2,2,2), mode='trilinear')
        self.up4 = nn.Upsample(scale_factor=(4,4,4), mode='trilinear')
        self.up8 = nn.Upsample(scale_factor=(8,8,8), mode='trilinear')
        self.up16 = nn.Upsample(scale_factor=(16,16,16), mode='trilinear')
        self.down2 = nn.MaxPool3d(kernel_size=(2,2,2))
        self.down4 = nn.MaxPool3d(kernel_size=(4,4,4))
        self.down8 = nn.MaxPool3d(kernel_size=(8,8,8))
        self.down16 = nn.MaxPool3d(kernel_size=(16,16,16))
        # input data conv
        self.conv_input_00 = nn.Conv3d(inchannels, f[0], kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        # 11*1*1 conv for answer
        self.conv_01_1 = nn.Conv3d(f[0]*5, 3, kernel_size=(1,1,1), stride=1, padding=(0,0,0))
        self.conv_11_1 = nn.Conv3d(f[0]*5, 3, kernel_size=(1,1,1), stride=1, padding=(0,0,0))
        self.conv_21_1 = nn.Conv3d(f[0]*5, 3, kernel_size=(1,1,1), stride=1, padding=(0,0,0))
        self.conv_31_1 = nn.Conv3d(f[0]*5, 3, kernel_size=(1,1,1), stride=1, padding=(0,0,0))
        self.conv_41_1 = nn.Conv3d(f[4], 3, kernel_size=(1,1,1), stride=1, padding=(0,0,0))
        # for x01
        self.res00 = Res_Bottle_Block(f[0],f[0],3,1,1)
        self.res10 = Res_Bottle_Block(f[0]*5,f[0],3,1,1)
        self.res20 = Res_Bottle_Block(f[0]*5,f[0],3,1,1)
        self.res30 = Res_Bottle_Block(f[0]*5,f[0],3,1,1)
        self.res40 = Res_Bottle_Block(f[4],f[0],3,1,1)
        # for x11
        self.res01 = Res_Bottle_Block(f[0],f[0],3,1,1)
        self.res11 = Res_Bottle_Block(f[1],f[0],3,1,1)
        self.res21 = Res_Bottle_Block(f[0]*5,f[0],3,1,1)
        self.res31 = Res_Bottle_Block(f[0]*5,f[0],3,1,1)
        self.res41 = Res_Bottle_Block(f[4],f[0],3,1,1)
        # for x21
        self.res02 = Res_Bottle_Block(f[0],f[0],3,1,1)
        self.res12 = Res_Bottle_Block(f[1],f[0],3,1,1)
        self.res22 = Res_Bottle_Block(f[2],f[0],3,1,1)
        self.res32 = Res_Bottle_Block(f[0]*5,f[0],3,1,1)
        self.res42 = Res_Bottle_Block(f[4],f[0],3,1,1)
        # for x31
        self.res03 = Res_Bottle_Block(f[0],f[0],3,1,1)
        self.res13 = Res_Bottle_Block(f[1],f[0],3,1,1)
        self.res23 = Res_Bottle_Block(f[2],f[0],3,1,1)
        self.res33 = Res_Bottle_Block(f[3],f[0],3,1,1)
        self.res43 = Res_Bottle_Block(f[4],f[0],3,1,1)
        # after cat
        self.res_01_0 = Res_Bottle_Block(f[0]*5,f[0]*5,3,1,1)
        self.res_11_0 = Res_Bottle_Block(f[0]*5,f[0]*5,3,1,1)
        self.res_21_0 = Res_Bottle_Block(f[0]*5,f[0]*5,3,1,1)
        self.res_31_0 = Res_Bottle_Block(f[0]*5,f[0]*5,3,1,1)
        # time resblock
        self.res_00_10 = self.Time_Resblocks(1,[f[0],f[1]],3,(2,2,2),1)# 3
        self.res_10_20 = self.Time_Resblocks(1,[f[1],f[2]],3,(2,2,2),1)# 4
        self.res_20_30 = self.Time_Resblocks(1,[f[2],f[3]],3,(2,2,2),1)# 6
        self.res_30_40 = self.Time_Resblocks(1,[f[3],f[4]],3,(2,2,2),1)# 3

        #upsample
        self.upscore5 = nn.Upsample(scale_factor=16, mode='nearest')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='nearest')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='nearest')

    def Time_Resblocks(self, num_layer, channels, kernel_size, stride, padding, bias=True):
        layers = []
        for i in range(num_layer):
            if i==0 :
                layers.append(Res_Bottle_Block_down(channels[0],channels[1],kernel_size=kernel_size,stride=stride,padding=padding,bias=bias))
            else :
                layers.append(Res_Bottle_Block(channels[1],channels[1],kernel_size=kernel_size,stride=1,padding=padding,bias=bias))
        return nn.Sequential(*layers)

    def MinMaxNorm(self, x, eps=1e-5):
        return (x-torch.min(x))/(torch.max(x)-torch.min(x)+eps)

    def forward(self, density, randomview=0):

        x00 = self.conv_input_00(density)# [2,64^3]->[f[0],64^3]
        x10 = self.res_00_10(x00)# [f[0],32^3]->[f[1],32^3]
        x20 = self.res_10_20(x10)# [f[1],16^3]->[f[2],16^3]
        x30 = self.res_20_30(x20)# [f[2],8^3]->[f[3],8^3]
        x40 = self.res_30_40(x30)# [f[3],4^3]->[f[4],4^3]

        x41 = x40
        x31 = self.res_31_0(torch.cat((self.res43(self.up2(x41)), self.res33(x30), self.res23(self.down2(x20)),
            self.res13(self.down4(x10)), self.res03(self.down8(x00))), 1))# [f[0]*5,11,32^2]; from [x41,x30,x20,x10,x00]
        x21 = self.res_21_0(torch.cat((self.res42(self.up4(x41)), self.res32(self.up2(x31)), self.res22(x20),
            self.res12(self.down2(x10)), self.res02(self.down4(x00))), 1))# [f[0]*5,11,64^2]; from [x41,x31,x20,x10,x00]
        x11 = self.res_11_0(torch.cat((self.res41(self.up8(x41)), self.res31(self.up4(x31)), self.res21(self.up2(x21)),
            self.res11(x10), self.res01(self.down2(x00))), 1))# [f[0]*5,11,128^2]; from [x41,x31,x21,x10,x00]
        x01 = self.res_01_0(torch.cat((self.res40(self.up16(x41)), self.res30(self.up8(x31)), self.res20(self.up4(x21)),
            self.res10(self.up2(x11)), self.res00(x00)), 1))# [f[0]*5,11,256^2]; from [x41,x31,x21,x11,x00]

        # short cut
        x01 = self.conv_01_1(x01)

        x11 = self.upscore2(x11)
        x11 = self.conv_11_1(x11)

        x21 = self.upscore3(x21)
        x21 = self.conv_21_1(x21)

        x31 = self.upscore4(x31)
        x31 = self.conv_31_1(x31)

        x41 = self.upscore5(x41)
        x41 = self.conv_41_1(x41)


        return x01, x11, x21, x31, x41