import torch
import torch.nn as nn
from torch.nn import (
    Conv2d,Linear,AvgPool2d,MaxPool2d,AdaptiveAvgPool2d,AdaptiveMaxPool2d,
    ReLU,GELU,Sigmoid,BatchNorm2d,LayerNorm
)
from ModelFactory.register import Registers
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

@Registers.block
class dartsv2_stem_k3(nn.Module):

    def __init__(self,in_channels,out_channels=None):
        super().__init__()
        self.out_channels = out_channels

        self.op1=CustomConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=1,groups=1)
        self.op2=BatchNorm2d(num_features=out_channels)

    def forward(self,x):
        B,C,H,W = x.shape
        out_channels = self.out_channels

        x = self.op1(x)
        x = self.op2(x)
        return x
