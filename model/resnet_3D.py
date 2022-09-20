# Modified from https://github.com/pytorch/vision/tree/master/torchvision/models/video

import torch
import torch.nn as nn
import math

__all__ = ['unet_18', 'unet_34']

useBias = False

class identity(nn.Module):

    def __init__(self , *args , **kwargs):

        super().__init__()
    
    def forward(self , x):
        return x

class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=useBias)

    @staticmethod
    def get_downsample_stride(stride , temporal_stride):
        if temporal_stride:
            return (temporal_stride, stride, stride)
        else:
            return (stride , stride , stride)

class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super().__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),#kernelsize=t*h*w，s同理
                padding=(1, 3, 3), bias=useBias),
            batchnorm(64),
            nn.ReLU(inplace=False))

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        if not isinstance(stride , int):
            temporal_stride , stride , stride = stride
        else:
            temporal_stride = stride

        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            # batchnorm(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(temporal_stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride , temporal_stride):
        if temporal_stride:
            return (temporal_stride, stride, stride)
        else:
            return (stride , stride , stride)

class GS3D(nn.Module):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1, ratio=2):
        if not isinstance(stride , int):
            temporal_stride , stride , stride = stride
        else:
            temporal_stride = stride

        super(GS3D, self).__init__()
      
        self.out_planes = out_planes
        self.midplanes = midplanes
        init_channels = math.ceil(midplanes / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv3d(in_planes, init_channels, kernel_size=(1,3,3), stride=(1,stride,stride), padding=(0,padding,padding), bias=False),
            batchnorm(init_channels),
            nn.ReLU(inplace=True)
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv3d(init_channels, new_channels, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,padding,padding), groups=init_channels, bias=False),
            batchnorm(new_channels),
            nn.ReLU(inplace=True)
        )

        self.temporalconv = nn.Sequential(
            nn.Conv3d(midplanes, out_planes, kernel_size=(3,1,1), stride=(temporal_stride,1,1), padding=(padding,0,0), bias=False),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        out = out[:,:self.midplanes,:,:]
        out = self.temporalconv(out)
        return out

    @staticmethod
    def get_downsample_stride(stride , temporal_stride):
        if temporal_stride:
            return (temporal_stride, stride, stride)
        else:
            return (stride , stride , stride)


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """
    def __init__(self):
        super().__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 1, 1), padding=(0, 3, 3),
                      bias=False),
            batchnorm(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            batchnorm(64),
            nn.ReLU(inplace=True))


class SEGating(nn.Module):

    def __init__(self , inplanes , reduction=16):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.attn_layer = nn.Sequential(
            nn.Conv3d(inplanes , inplanes , kernel_size=1 , stride=1 , bias=True),
            nn.Sigmoid()
        )
        
    def forward(self , x):

        out = self.pool(x)
        y = self.attn_layer(out)
        return x * y


'''class SEGating(nn.Module):#FG

    def __init__(self , inplanes , reduction=16):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.conv = nn.Sequential(
            nn.Conv3d(inplanes * 2 , inplanes , kernel_size=1 , stride=1 , bias=True),
        )
        self.attn_layer = nn.Sequential(
            nn.Conv3d(inplanes , inplanes , kernel_size=1 , stride=1 , bias=True),
            nn.Sigmoid()
        )
        
    def forward(self , x):

        out1 = self.avg_pool(x)
        out2 = self.max_pool(x)
        out = torch.cat((out1, out2), 1)
        out = self.conv(out)
        y = self.attn_layer(out)
        return x * y'''

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(in_planes=inplanes,out_planes=planes,midplanes=midplanes,stride=stride,padding=1,ratio=2),
            batchnorm(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(in_planes=planes,out_planes=planes,midplanes=midplanes,stride=1,padding=1,ratio=2),
            batchnorm(planes)
        )
        self.fg = SEGating(planes) ## Feature Gating
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.fg(out)

        return out

class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1 )
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2 , temporal_stride=1)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2 , temporal_stride=1)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2, temporal_stride=1)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x_0 = self.stem(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        return x_0 , x_1 , x_2 , x_3 , x_4

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, temporal_stride=None):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride , temporal_stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                batchnorm(planes * block.expansion)
            )
            stride = ds_stride

        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample ))

        self.inplanes = planes * block.expansion
        #for i in range(1, blocks):
         #   layers.append(block(self.inplanes, planes, conv_builder ))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _video_resnet(arch, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)
    ## TODO: Other 3D resnet models, like S3D, r(2+1)D.

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def unet_18(pretrained=False, bn=False, progress=True, **kwargs):
    """
    Construct 18 layer Unet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 encoder
    """
    global batchnorm
    if bn:
        batchnorm = nn.BatchNorm3d
    else:
        batchnorm = identity

    return _video_resnet('r3d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[GS3D] * 4,
                         layers=[1, 1, 1, 1],
                         stem=R2Plus1dStem, **kwargs)

def unet_34(pretrained=False, bn=False, progress=True, **kwargs):
    """
    Construct 34 layer Unet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 encoder
    """
    global batchnorm
    # bn = False
    if bn:
        batchnorm = nn.BatchNorm3d
    else:
        batchnorm = identity


    return _video_resnet('r3d_34',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[3, 4, 6, 3],
                         stem=BasicStem, **kwargs)
