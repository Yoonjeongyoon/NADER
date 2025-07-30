import torch
import torch.nn as nn
from torch.nn import (
    Conv2d,Linear,AvgPool2d,MaxPool2d,AdaptiveAvgPool2d,AdaptiveMaxPool2d,
    ReLU,GELU,Sigmoid,BatchNorm2d,LayerNorm
)
from ModelFactory.register import Registers
from timm.models.layers import trunc_normal_, DropPath
from typing import *
import math

class CustomConv2d(nn.Module):

    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride:Union[int, Tuple[int, int]] = 1,
        dilation:Union[int, Tuple[int, int]] = 1,
        **kwargs):
        super().__init__()
        if isinstance(kernel_size,int):
            kernel_size=(kernel_size,kernel_size)
        if isinstance(stride,int):
            stride=(stride,stride)
        if isinstance(dilation,int):
            dilation=(dilation,dilation)
        self.padding_custom=((dilation[0]*(kernel_size[0]-1)+1-stride[0])/2,(dilation[1]*(kernel_size[1]-1)+1-stride[1])/2)
        self.padding_ceil = (max(math.ceil(self.padding_custom[0]),0),max(math.ceil(self.padding_custom[1]),0))
        self.padding_right = ((dilation[0]*(kernel_size[0]-1)+1)/2,(dilation[1]*(kernel_size[1]-1)+1)/2)
        self.conv2d = Conv2d(in_channels,out_channels,kernel_size,stride,self.padding_ceil,dilation,**kwargs)
    
    def forward(self, input):
        res = self.conv2d(input)
        if self.padding_ceil[0]==self.padding_right[0]:
            res =  res[:,:,:-1]
        if self.padding_ceil[1]==self.padding_right[1]:
            res =  res[:,:,:,:-1]
        return res

class CustomMaxPool2d(nn.Module):

    def __init__(self,
        kernel_size: Union[int, Tuple[int, int]],
        stride:Union[int, Tuple[int, int]] = 1,
        **kwargs):
        super().__init__()
        if isinstance(kernel_size,int):
            kernel_size=(kernel_size,kernel_size)
        if isinstance(stride,int):
            stride=(stride,stride)
        self.padding_custom=((kernel_size[0]-stride[0])/2,(kernel_size[1]-stride[1])/2)
        self.padding_ceil = (max(math.ceil(self.padding_custom[0]),0),max(math.ceil(self.padding_custom[1]),0))
        self.padding_right = (kernel_size[0]/2,kernel_size[1]/2)
        self.pool = MaxPool2d(kernel_size,stride,self.padding_ceil,**kwargs)

    def forward(self, input):
        res = self.pool(input)
        if self.padding_ceil[0]==self.padding_right[0]:
            res =  res[:,:,:-1]
        if self.padding_ceil[1]==self.padding_right[1]:
            res =  res[:,:,:,:-1]
        return res

class CustomAvgPool2d(nn.Module):

    def __init__(self,
        kernel_size: Union[int, Tuple[int, int]],
        stride:Union[int, Tuple[int, int]] = 1,
        **kwargs):
        super().__init__()
        if isinstance(kernel_size,int):
            kernel_size=(kernel_size,kernel_size)
        if isinstance(stride,int):
            stride=(stride,stride)
        self.padding_custom=((kernel_size[0]-stride[0])/2,(kernel_size[1]-stride[1])/2)
        self.padding_ceil = (max(math.ceil(self.padding_custom[0]),0),max(math.ceil(self.padding_custom[1]),0))
        self.padding_right = (kernel_size[0]/2,kernel_size[1]/2)
        self.pool = AvgPool2d(kernel_size,stride,self.padding_ceil,**kwargs)

    def forward(self, input):
        res = self.pool(input)
        if self.padding_ceil[0]==self.padding_right[0]:
            res =  res[:,:,:-1]
        if self.padding_ceil[1]==self.padding_right[1]:
            res =  res[:,:,:,:-1]
        return res

class convnext_trail3_convnext_p3508_p700_base(nn.Module):

    def __init__(self,in_channels,out_channels=None,drop_path=0.0,layer_scale_init_value=1e-6):
        super().__init__()
        self.out_channels = out_channels

        self.op1=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=7,stride=1,dilation=1,groups=in_channels)
        self.op2=LayerNorm(normalized_shape=(in_channels)//(1))
        self.op3=Linear(in_features=(in_channels)//(1),out_features=(in_channels*4)//(1))
        self.op4=GELU()
        self.op5=Linear(in_features=((in_channels*4)//(1))//(1),out_features=(in_channels)//(1))
        self.op6=AdaptiveAvgPool2d(output_size=1)
        self.op7=CustomConv2d(in_channels=(in_channels)//(1),out_channels=in_channels//16,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op8=ReLU()
        self.op9=CustomConv2d(in_channels=in_channels//16,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op10=Sigmoid()
        self.op11=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,dilation=1,groups=in_channels)
        self.op12=ReLU()
        self.op13=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op14=Sigmoid()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channels)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self,x):
        B,in_channels,H,W = x.shape
        out_channels = self.out_channels

        h1 = self.op1(x)
        h2 = self.op11(x)
        h1 = h1.permute(0,2,3,1)
        h2 = self.op12(h2)
        h1 = self.op2(h1)
        h2 = self.op13(h2)
        h1 = self.op3(h1)
        h2 = self.op14(h2)
        h1 = self.op4(h1)
        h2 = h2*x
        h1 = self.op5(h1)
        if self.gamma is not None:
            h1 = self.gamma * h1
        h1 = h1.permute(0,3,1,2)
        h3 = self.op6(h1)
        h3 = self.op7(h3)
        h3 = self.op8(h3)
        h3 = self.op9(h3)
        h3 = self.op10(h3)
        h3 = h3*h1
        h2 = h2+h3
        h2 = x+self.drop_path(h2)
        return h2

class convnext_trail3_convnext_p207_stem(nn.Module):

    def __init__(self,in_channels,out_channels=None):
        super().__init__()
        self.out_channels = out_channels

        self.op1=CustomConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=2,dilation=1,groups=1)
        self.op2=BatchNorm2d(num_features=out_channels)
        self.op3=ReLU()
        self.op4=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=2,dilation=1,groups=1)
        self.op5=BatchNorm2d(num_features=out_channels)
        self.op6=ReLU()

    def forward(self,x):
        B,in_channels,H,W = x.shape
        out_channels = self.out_channels

        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        x = self.op4(x)
        x = self.op5(x)
        x = self.op6(x)
        return x

class convnext_trail3_convnext_p1380_downsample(nn.Module):

    def __init__(self,in_channels,out_channels=None):
        super().__init__()
        self.out_channels = out_channels

        self.op1=CustomConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=2,stride=2,dilation=1,groups=1)
        self.op2=BatchNorm2d(num_features=out_channels)
        self.op3=ReLU()

    def forward(self,x):
        B,in_channels,H,W = x.shape
        out_channels = self.out_channels

        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        return x

@Registers.model
class convnext_trail3_convnext_p3508_p700_2282(nn.Module):

    def __init__(self,num_classes=100) -> None:
        super().__init__()
        block = convnext_trail3_convnext_p3508_p700_base
        stem = convnext_trail3_convnext_p207_stem
        downsample = convnext_trail3_convnext_p1380_downsample
        dp_rates=[x.item() for x in torch.linspace(0, 0.2, 14)]
        dim = 105
        self.stem = stem(in_channels=3,out_channels=dim)
        layers = []
        for i in range(2):
            layers.append(block(in_channels=dim,out_channels=dim,drop_path=dp_rates[i]))
        self.layer1 = nn.Sequential(*layers)
        self.downsample1 = downsample(in_channels=dim,out_channels=dim*2)
        layers = []
        for i in range(2):
            layers.append(block(in_channels=dim*2,out_channels=dim*2,drop_path=dp_rates[2+i]))
        self.layer2 = nn.Sequential(*layers)
        self.downsample2 = downsample(in_channels=dim*2,out_channels=dim*4)
        layers = []
        for i in range(8):
            layers.append(block(in_channels=dim*4,out_channels=dim*4,drop_path=dp_rates[4+i]))
        self.layer3 = nn.Sequential(*layers)
        self.downsample3 = downsample(in_channels=dim*4,out_channels=dim*8)
        layers = []
        for i in range(2):
            layers.append(block(in_channels=dim*8,out_channels=dim*8,drop_path=dp_rates[12+i]))
        self.layer4 = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim*8,num_classes)

        self.apply(self._init_weights)
        self.fc.weight.data.mul_(1e-6)
        self.fc.bias.data.mul_(1e-6)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self,x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.downsample1(h)
        h = self.layer2(h)
        h = self.downsample2(h)
        h = self.layer3(h)
        h = self.downsample3(h)
        h = self.layer4(h)
        h = self.avg_pool(h)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        return h

