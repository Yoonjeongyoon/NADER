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

class resnet_basic_seed777_resnet_basic_p2703_p4472_p3537_p3605_p3869_base(nn.Module):

    def __init__(self,in_channels,out_channels=None):
        super().__init__()
        self.out_channels = out_channels

        self.op1=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,dilation=1,groups=1)
        self.op2=BatchNorm2d(num_features=in_channels)
        self.op3=ReLU()
        self.op4=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,dilation=1,groups=1)
        self.op5=BatchNorm2d(num_features=in_channels)
        self.op6=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op7=BatchNorm2d(num_features=in_channels)
        self.op8=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op9=BatchNorm2d(num_features=in_channels)
        self.op10=ReLU()
        self.op11=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op12=BatchNorm2d(num_features=in_channels)
        self.op13=ReLU()
        self.op14=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op15=BatchNorm2d(num_features=in_channels)
        self.op16=AdaptiveAvgPool2d(output_size=1)
        self.op17=Linear(in_features=in_channels,out_features=in_channels)
        self.op18=ReLU()
        self.op19=Linear(in_features=in_channels,out_features=in_channels)
        self.op20=Sigmoid()
        self.op21=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op22=BatchNorm2d(num_features=in_channels)
        self.op23=ReLU()
        self.op24=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op25=BatchNorm2d(num_features=in_channels)
        self.op26=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op27=BatchNorm2d(num_features=in_channels)
        self.op28=ReLU()
        self.op29=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op30=BatchNorm2d(num_features=in_channels)
        self.op31=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op32=BatchNorm2d(num_features=in_channels)
        self.op33=ReLU()
        self.op34=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op35=BatchNorm2d(num_features=in_channels)

    def forward(self,x):
        B,in_channels,H,W = x.shape
        out_channels = self.out_channels

        h1 = self.op1(x)
        h2 = self.op6(x)
        h3 = self.op8(x)
        h1 = self.op2(h1)
        h2 = self.op7(h2)
        h3 = self.op9(h3)
        h1 = self.op3(h1)
        h1 = self.op4(h1)
        h1 = self.op5(h1)
        h1 = h1+h2
        h1 = self.op10(h1)
        h3 = h3+h1
        h3 = self.op11(h3)
        h3 = self.op12(h3)
        h3 = self.op13(h3)
        h3 = self.op14(h3)
        h3 = self.op15(h3)
        h4 = x+h3
        h5 = self.op16(h4)
        h5 = h5.reshape(B,in_channels)
        h5 = self.op17(h5)
        h5 = self.op18(h5)
        h5 = self.op19(h5)
        h5 = self.op20(h5)
        h5 = h5.reshape(B,in_channels,1,1)
        h5 = h5*h4
        h5 = self.op21(h5)
        h5 = self.op22(h5)
        h5 = self.op23(h5)
        h5 = self.op24(h5)
        h5 = self.op25(h5)
        h5 = h5+h4
        h5 = self.op26(h5)
        h5 = self.op27(h5)
        h5 = self.op28(h5)
        h5 = self.op29(h5)
        h5 = self.op30(h5)
        h5 = h5+h4
        h6 = self.op31(h5)
        h6 = self.op32(h6)
        h6 = self.op33(h6)
        h6 = self.op34(h6)
        h6 = self.op35(h6)
        h6 = h6+h5
        return h6

class resnet_basic_seed777_resnet_basic_p2703_p4472_p3537_p3605_p3869_base_dp(nn.Module):

    def __init__(self,in_channels,out_channels=None,drop_path=0.0):
        super().__init__()
        self.out_channels = out_channels

        self.op1=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,dilation=1,groups=1)
        self.op2=BatchNorm2d(num_features=in_channels)
        self.op3=ReLU()
        self.op4=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,dilation=1,groups=1)
        self.op5=BatchNorm2d(num_features=in_channels)
        self.op6=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op7=BatchNorm2d(num_features=in_channels)
        self.op8=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op9=BatchNorm2d(num_features=in_channels)
        self.op10=ReLU()
        self.op11=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op12=BatchNorm2d(num_features=in_channels)
        self.op13=ReLU()
        self.op14=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op15=BatchNorm2d(num_features=in_channels)
        self.op16=AdaptiveAvgPool2d(output_size=1)
        self.op17=Linear(in_features=in_channels,out_features=in_channels)
        self.op18=ReLU()
        self.op19=Linear(in_features=in_channels,out_features=in_channels)
        self.op20=Sigmoid()
        self.op21=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op22=BatchNorm2d(num_features=in_channels)
        self.op23=ReLU()
        self.op24=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op25=BatchNorm2d(num_features=in_channels)
        self.op26=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op27=BatchNorm2d(num_features=in_channels)
        self.op28=ReLU()
        self.op29=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op30=BatchNorm2d(num_features=in_channels)
        self.op31=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op32=BatchNorm2d(num_features=in_channels)
        self.op33=ReLU()
        self.op34=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op35=BatchNorm2d(num_features=in_channels)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,x):
        B,in_channels,H,W = x.shape
        out_channels = self.out_channels

        h1 = self.op1(x)
        h2 = self.op6(x)
        h3 = self.op8(x)
        h1 = self.op2(h1)
        h2 = self.op7(h2)
        h3 = self.op9(h3)
        h1 = self.op3(h1)
        h1 = self.op4(h1)
        h1 = self.op5(h1)
        h1 = h1+h2
        h1 = self.op10(h1)
        h3 = h3+h1
        h3 = self.op11(h3)
        h3 = self.op12(h3)
        h3 = self.op13(h3)
        h3 = self.op14(h3)
        h3 = self.op15(h3)
        h4 = x+self.drop_path(h3)
        h5 = self.op16(h4)
        h5 = h5.reshape(B,in_channels)
        h5 = self.op17(h5)
        h5 = self.op18(h5)
        h5 = self.op19(h5)
        h5 = self.op20(h5)
        h5 = h5.reshape(B,in_channels,1,1)
        h5 = h5*h4
        h5 = self.op21(h5)
        h5 = self.op22(h5)
        h5 = self.op23(h5)
        h5 = self.op24(h5)
        h5 = self.op25(h5)
        h5 = self.drop_path(h5)+h4
        h5 = self.op26(h5)
        h5 = self.op27(h5)
        h5 = self.op28(h5)
        h5 = self.op29(h5)
        h5 = self.op30(h5)
        h5 = self.drop_path(h5)+h4
        h6 = self.op31(h5)
        h6 = self.op32(h6)
        h6 = self.op33(h6)
        h6 = self.op34(h6)
        h6 = self.op35(h6)
        h6 = self.drop_path(h6)+h5
        return h6

class resnet_basic_seed777_resnet_basic_p2703_p4472_p3537_p3605_p3869_base_dp2(nn.Module):

    def __init__(self,in_channels,out_channels=None,drop_path=0.0):
        super().__init__()
        self.out_channels = out_channels

        self.op1=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,dilation=1,groups=1)
        self.op2=BatchNorm2d(num_features=in_channels)
        self.op3=ReLU()
        self.op4=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,dilation=1,groups=1)
        self.op5=BatchNorm2d(num_features=in_channels)
        self.op6=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op7=BatchNorm2d(num_features=in_channels)
        self.op8=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op9=BatchNorm2d(num_features=in_channels)
        self.op10=ReLU()
        self.op11=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op12=BatchNorm2d(num_features=in_channels)
        self.op13=ReLU()
        self.op14=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op15=BatchNorm2d(num_features=in_channels)
        self.op16=AdaptiveAvgPool2d(output_size=1)
        self.op17=Linear(in_features=in_channels,out_features=in_channels)
        self.op18=ReLU()
        self.op19=Linear(in_features=in_channels,out_features=in_channels)
        self.op20=Sigmoid()
        self.op21=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op22=BatchNorm2d(num_features=in_channels)
        self.op23=ReLU()
        self.op24=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op25=BatchNorm2d(num_features=in_channels)
        self.op26=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op27=BatchNorm2d(num_features=in_channels)
        self.op28=ReLU()
        self.op29=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op30=BatchNorm2d(num_features=in_channels)
        self.op31=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op32=BatchNorm2d(num_features=in_channels)
        self.op33=ReLU()
        self.op34=CustomConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op35=BatchNorm2d(num_features=in_channels)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,x):
        B,in_channels,H,W = x.shape
        out_channels = self.out_channels

        h1 = self.op1(x)
        h2 = self.op6(x)
        h3 = self.op8(x)
        h1 = self.op2(h1)
        h2 = self.op7(h2)
        h3 = self.op9(h3)
        h1 = self.op3(h1)
        h1 = self.op4(h1)
        h1 = self.op5(h1)
        h1 = h1+h2
        h1 = self.op10(h1)
        h3 = h3+h1
        h3 = self.op11(h3)
        h3 = self.op12(h3)
        h3 = self.op13(h3)
        h3 = self.op14(h3)
        h3 = self.op15(h3)
        h4 = x+h3
        h5 = self.op16(h4)
        h5 = h5.reshape(B,in_channels)
        h5 = self.op17(h5)
        h5 = self.op18(h5)
        h5 = self.op19(h5)
        h5 = self.op20(h5)
        h5 = h5.reshape(B,in_channels,1,1)
        h5 = h5*h4
        h5 = self.op21(h5)
        h5 = self.op22(h5)
        h5 = self.op23(h5)
        h5 = self.op24(h5)
        h5 = self.op25(h5)
        h5 = h5+h4
        h5 = self.op26(h5)
        h5 = self.op27(h5)
        h5 = self.op28(h5)
        h5 = self.op29(h5)
        h5 = self.op30(h5)
        h5 = h5+h4
        h6 = self.op31(h5)
        h6 = self.op32(h6)
        h6 = self.op33(h6)
        h6 = self.op34(h6)
        h6 = self.op35(h6)
        h6 = h6+h5
        return h6

class resnet_basic_stem(nn.Module):

    def __init__(self,in_channels,out_channels=None):
        super().__init__()
        self.out_channels = out_channels

        self.op1=CustomConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=1,groups=1)
        self.op2=BatchNorm2d(num_features=out_channels)
        self.op3=ReLU()

    def forward(self,x):
        B,in_channels,H,W = x.shape
        out_channels = self.out_channels

        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        return x

class resnet_basic_downsample(nn.Module):

    def __init__(self,in_channels,out_channels=None):
        super().__init__()
        self.out_channels = out_channels

        self.op1=CustomConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=2,dilation=1,groups=1)
        self.op2=BatchNorm2d(num_features=out_channels)
        self.op3=ReLU()
        self.op4=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=1,groups=1)
        self.op5=BatchNorm2d(num_features=out_channels)
        self.op6=ReLU()
        self.op7=CustomAvgPool2d(kernel_size=2,stride=2)
        self.op8=CustomConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,dilation=1,groups=1)

    def forward(self,x):
        B,in_channels,H,W = x.shape
        out_channels = self.out_channels

        h1 = self.op1(x)
        h2 = self.op7(x)
        h1 = self.op2(h1)
        h2 = self.op8(h2)
        h1 = self.op3(h1)
        h1 = self.op4(h1)
        h1 = self.op5(h1)
        h2 = h2+h1
        h2 = self.op6(h2)
        return h2

@Registers.model
class resnet_p2703_p4472_p3537_p3605_p3869(nn.Module):

    def __init__(self,num_classes=100) -> None:
        super().__init__()
        block = resnet_basic_seed777_resnet_basic_p2703_p4472_p3537_p3605_p3869_base
        stem = resnet_basic_stem
        downsample = resnet_basic_downsample
        dim = 29
        self.stem = stem(in_channels=3,out_channels=dim)
        layers = []
        for _ in range(5):
            layers.append(block(in_channels=dim,out_channels=dim))
        self.layer1 = nn.Sequential(*layers)
        self.downsample1 = downsample(in_channels=dim,out_channels=dim*2)
        layers = []
        for _ in range(5):
            layers.append(block(in_channels=dim*2,out_channels=dim*2))
        self.layer2 = nn.Sequential(*layers)
        self.downsample2 = downsample(in_channels=dim*2,out_channels=dim*4)
        layers = []
        for _ in range(5):
            layers.append(block(in_channels=dim*4,out_channels=dim*4))
        self.layer3 = nn.Sequential(*layers)
        layers = []
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim*4,num_classes)

    def forward(self,x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.downsample1(h)
        h = self.layer2(h)
        h = self.downsample2(h)
        h = self.layer3(h)
        h = self.avg_pool(h)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        return h

class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=False),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=False)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


@Registers.model
class resnet_p2703_p4472_p3537_p3605_p3869_dp(nn.Module):

    def __init__(self,num_classes=100) -> None:
        super().__init__()
        block = resnet_basic_seed777_resnet_basic_p2703_p4472_p3537_p3605_p3869_base_dp
        stem = resnet_basic_stem
        downsample = resnet_basic_downsample
        dim = 29
        dp_rates=[x.item() for x in torch.linspace(0, 0.2, 15)] 
        self.stem = stem(in_channels=3,out_channels=dim)
        layers = []
        for i in range(5):
            layers.append(block(in_channels=dim,out_channels=dim,drop_path=dp_rates[i]))
        self.layer1 = nn.Sequential(*layers)
        self.downsample1 = downsample(in_channels=dim,out_channels=dim*2)
        layers = []
        for i in range(5):
            layers.append(block(in_channels=dim*2,out_channels=dim*2,drop_path=dp_rates[5+i]))
        self.layer2 = nn.Sequential(*layers)
        self.downsample2 = downsample(in_channels=dim*2,out_channels=dim*4)
        layers = []
        for i in range(5):
            layers.append(block(in_channels=dim*4,out_channels=dim*4,drop_path=dp_rates[10+i]))
        self.layer3 = nn.Sequential(*layers)
        layers = []
        self.auxiliary_head = AuxiliaryHeadCIFAR(dim*4, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim*4,num_classes)

    #     self.apply(self._init_weights)
    #     self.fc.weight.data.mul_(1e-6)
    #     self.fc.bias.data.mul_(1e-6)

    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         trunc_normal_(m.weight, std=.02)
    #         # nn.init.constant_(m.bias, 0)

    def forward(self,x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.downsample1(h)
        h = self.layer2(h)
        h = self.downsample2(h)
        if self.training:
            logits_aux = self.auxiliary_head(h)
        h1 = self.layer3(h)
        h1 = self.avg_pool(h1)
        h1 = h1.view(h1.size(0), -1)
        h1 = self.fc(h1)
        if self.training:
            return h1,logits_aux
        else:
            return h1

