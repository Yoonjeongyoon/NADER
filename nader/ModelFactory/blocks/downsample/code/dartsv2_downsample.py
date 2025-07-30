import torch
import torch.nn as nn
from torch.autograd import Variable
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

class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out

def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        # mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)).to(x.device)
        x.div_(keep_prob)
        x.mul_(mask)
    return x

@Registers.block
class dartsv2_downsample(nn.Module):

    def __init__(self,in_channels,out_channels,C_pre=None,reduction_pre=False):
        super().__init__()
        self.out_channels = out_channels
        if C_pre==None:
            C_pre = in_channels
        if reduction_pre:
            self.preprocess0 = FactorizedReduce(C_pre, out_channels)
        elif C_pre!=out_channels:
            self.preprocess0 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_pre, out_channels, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.preprocess0 = nn.Identity()
        if in_channels!=out_channels:
            self.preprocess1 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.preprocess1 = nn.Identity()
        in_channels = out_channels

        self.op1=CustomMaxPool2d(kernel_size=2,stride=2)
        self.op2=CustomMaxPool2d(kernel_size=2,stride=2)
        self.op3=CustomMaxPool2d(kernel_size=2,stride=2)
        self.op4=CustomMaxPool2d(kernel_size=2,stride=2)
        self.op5=CustomMaxPool2d(kernel_size=2,stride=2)
        self.op6=ReLU()
        self.op7=CustomConv2d(in_channels=(((in_channels)+(in_channels))+(in_channels))+(in_channels),out_channels=out_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op8=BatchNorm2d(num_features=out_channels)
        
    def forward(self,x1,x2=None):
        if x2==None:
            x2=x1
        x1 = self.preprocess0(x1)
        x2 = self.preprocess1(x2)

        h1 = self.op1(x1)
        h2 = self.op4(x1)
        h3 = self.op2(x2)
        h4 = self.op3(x2)
        h5 = self.op5(x2)
        if self.training:
            h1=drop_path(h1,0.2)
        if self.training:
            h2=drop_path(h2,0.2)
        if self.training:
            h3=drop_path(h3,0.2)
        if self.training:
            h4=drop_path(h4,0.2)
        if self.training:
            h5=drop_path(h5,0.2)
        h3 = h3+h1
        h4 = h4+h3
        h6 = h3+h2
        h7 = h3+h5
        h8 = torch.concat([h3,h4,h6,h7], dim=1)
        h8 = self.op6(h8)
        h8 = self.op7(h8)
        h8 = self.op8(h8)
        return h8
