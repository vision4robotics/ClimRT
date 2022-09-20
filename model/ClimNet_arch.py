import math
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_3D import SEGating

def joinTensors(X1 , X2 , type="concat"):

    if type == "concat":
        return torch.cat([X1 , X2] , dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1


class Conv_2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if batchnorm:
            self.conv += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)

class upConv3D(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose" , batchnorm=False):

        super().__init__()

        self.upmode = upmode

        if self.upmode=="transpose":
            self.upconv = nn.ModuleList(
                [nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                SEGating(out_ch)
                ]
            )

        else:
            self.upconv = nn.ModuleList(
                [nn.Upsample(mode='trilinear', scale_factor=(1,2,2), align_corners=False),
                nn.Conv3d(in_ch, out_ch , kernel_size=1 , stride=1),
                SEGating(out_ch)
                ]
            )

        if batchnorm:
            self.upconv += [nn.BatchNorm3d(out_ch)]

        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):

        return self.upconv(x)

class Conv_3d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    SEGating(out_ch)
                    ]

        if batchnorm:
            self.conv += [nn.BatchNorm3d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)

class upConv2D(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose" , batchnorm=False):

        super().__init__()

        self.upmode = upmode

        if self.upmode=="transpose":
            self.upconv = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)]

        else:
            self.upconv = [
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_ch, out_ch , kernel_size=1 , stride=1)
            ]

        if batchnorm:
            self.upconv += [nn.BatchNorm2d(out_ch)]

        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):

        return self.upconv(x)

class MSfusion(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, batchnorm=False):
        super(MSfusion, self).__init__()
        self.batchnorm =batchnorm
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.bn3d = nn.BatchNorm3d(self.out_ch)
        self.upconv = upConv3D(self.in_ch, self.out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace = False)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Sequential(
                nn.Conv3d(self.in_ch, self.out_ch,  kernel_size=(1,1,1), stride=1),
                )
        self.fg = SEGating(self.out_ch)

    def forward(self, x):
        xm = x
        x = self.upconv(x)
        x = self.relu(x)
        xm = self.avg_pool(xm)
        xm = self.conv(xm)
        xm = torch.sigmoid(xm)
        x = x*xm
        x = self.fg(x)
        if self.batchnorm:
            x = self.bn3d(x)

        return x


class UNet_3D_3D(nn.Module):
    def __init__(self, block , n_inputs, n_outputs, batchnorm=False , joinType="concat"):
        super().__init__()

        nf = [512 , 256 , 128 , 64]        
        out_channels = 3*n_outputs
        self.joinType = joinType
        self.n_outputs = n_outputs

        growth = 2 if joinType == "concat" else 1
        self.lrelu = nn.LeakyReLU(0.2, True)

        unet_3D = importlib.import_module(".resnet_3D" , "model")
        if n_outputs > 1:
            unet_3D.useBias = True
        self.encoder = getattr(unet_3D , block)(pretrained=False , bn=batchnorm)     #？？       

        
        '''self.decoder = nn.Sequential(
            Conv_3d(nf[0], nf[0] , kernel_size=(3,3,3), padding=(1,1,1) bias=True, batchnorm=batchnorm),
            MSfusion(nf[0]*growth, nf[1], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1), batchnorm=batchnorm),
            MSfusion(nf[1]*growth, nf[2], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1), batchnorm=batchnorm),
            MSfusion(nf[2]*growth, nf[3], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1), batchnorm=batchnorm),
            Conv_3d(nf[3]*growth, nf[3], kernel_size=(3,3,3), padding=(1,1,1), bias=True, batchnorm=batchnorm)
        )'''

        self.decoder = nn.Sequential(
            Conv_3d(nf[0], nf[0] , kernel_size=(3,3,3), padding=(1,1,1), bias=True, batchnorm=batchnorm),
            MSfusion(nf[0]*growth, nf[1], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , batchnorm=batchnorm),
            MSfusion(nf[1]*growth, nf[2], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , batchnorm=batchnorm),
            MSfusion(nf[2]*growth, nf[3], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , batchnorm=batchnorm),
            Conv_3d(nf[3]*growth , nf[3], kernel_size=(3,3,3), padding=(1,1,1), bias=True, batchnorm=batchnorm)
        )

        self.feature_fuse = Conv_2d(nf[3]*n_inputs , nf[3] , kernel_size=1 , stride=1, batchnorm=batchnorm)

        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf[3], out_channels , kernel_size=7 , stride=1, padding=0) 
        )         

    def forward(self, images):

        images = torch.stack(images , dim=2)

        ## Batch mean normalization works slightly better than global mean normalization, thanks to https://github.com/myungsub/CAIN
        mean_ = images.mean(2, keepdim=True).mean(3, keepdim=True).mean(4,keepdim=True)
        images = images-mean_ 

        x_0 , x_1 , x_2 , x_3 , x_4 = self.encoder(images)

        dx_3 = self.lrelu(self.decoder[0](x_4))
        dx_3 = joinTensors(dx_3 , x_4 , type=self.joinType)

        dx_2 = self.lrelu(self.decoder[1](dx_3))
        dx_2 = joinTensors(dx_2 , x_3 , type=self.joinType)

        dx_1 = self.lrelu(self.decoder[2](dx_2))
        dx_1 = joinTensors(dx_1 , x_2 , type=self.joinType)

        dx_0 = self.lrelu(self.decoder[3](dx_1))
        dx_0 = joinTensors(dx_0 , x_1 , type=self.joinType)

        dx_out = self.lrelu(self.decoder[4](dx_0))
        dx_out = torch.cat(torch.unbind(dx_out , 2) , 1)

        out = self.lrelu(self.feature_fuse(dx_out))
        out = self.outconv(out)

        out = torch.split(out, dim=1, split_size_or_sections=3)
        mean_ = mean_.squeeze(2)
        out = [o+mean_ for o in out]
        return out

