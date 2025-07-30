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
class dartsv2_base(nn.Module):

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

        self.op1=ReLU()
        self.op2=CustomConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=1,groups=out_channels)
        self.op3=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op4=BatchNorm2d(num_features=out_channels)
        self.op5=ReLU()
        self.op6=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=1,groups=out_channels)
        self.op7=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op8=BatchNorm2d(num_features=out_channels)
        self.op9=ReLU()
        self.op10=CustomConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=1,groups=out_channels)
        self.op11=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op12=BatchNorm2d(num_features=out_channels)
        self.op13=ReLU()
        self.op14=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=1,groups=out_channels)
        self.op15=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op16=BatchNorm2d(num_features=out_channels)
        self.op17=ReLU()
        self.op18=CustomConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=1,groups=out_channels)
        self.op19=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op20=BatchNorm2d(num_features=out_channels)
        self.op21=ReLU()
        self.op22=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=1,groups=out_channels)
        self.op23=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op24=BatchNorm2d(num_features=out_channels)
        self.op25=ReLU()
        self.op26=CustomConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=1,groups=out_channels)
        self.op27=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op28=BatchNorm2d(num_features=out_channels)
        self.op29=ReLU()
        self.op30=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=1,groups=out_channels)
        self.op31=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op32=BatchNorm2d(num_features=out_channels)
        self.op33=ReLU()
        self.op34=CustomConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=1,groups=out_channels)
        self.op35=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op36=BatchNorm2d(num_features=out_channels)
        self.op37=ReLU()
        self.op38=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=1,groups=out_channels)
        self.op39=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op40=BatchNorm2d(num_features=out_channels)
        self.op41=ReLU()
        self.op42=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,dilation=2,groups=out_channels)
        self.op43=CustomConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op44=BatchNorm2d(num_features=out_channels)
        self.op45=ReLU()
        self.op46=CustomConv2d(in_channels=(((out_channels)+(out_channels))+(in_channels))+(in_channels),out_channels=out_channels,kernel_size=1,stride=1,dilation=1,groups=1)
        self.op47=BatchNorm2d(num_features=out_channels)
        
    def forward(self,x1,x2=None):
        if x2==None:
            x2=x1
        x1 = self.preprocess0(x1)
        x2 = self.preprocess1(x2)

        h1 = self.op1(x1)
        h2 = self.op17(x1)
        h3 = self.op9(x2)
        h4 = self.op25(x2)
        h5 = self.op33(x2)
        h1 = self.op2(h1)
        h2 = self.op18(h2)
        h3 = self.op10(h3)
        h4 = self.op26(h4)
        h5 = self.op34(h5)
        h1 = self.op3(h1)
        h2 = self.op19(h2)
        h3 = self.op11(h3)
        h4 = self.op27(h4)
        h5 = self.op35(h5)
        h1 = self.op4(h1)
        h2 = self.op20(h2)
        h3 = self.op12(h3)
        h4 = self.op28(h4)
        h5 = self.op36(h5)
        h1 = self.op5(h1)
        h2 = self.op21(h2)
        h3 = self.op13(h3)
        h4 = self.op29(h4)
        h5 = self.op37(h5)
        h1 = self.op6(h1)
        h2 = self.op22(h2)
        h3 = self.op14(h3)
        h4 = self.op30(h4)
        h5 = self.op38(h5)
        h1 = self.op7(h1)
        h2 = self.op23(h2)
        h3 = self.op15(h3)
        h4 = self.op31(h4)
        h5 = self.op39(h5)
        h1 = self.op8(h1)
        h2 = self.op24(h2)
        h3 = self.op16(h3)
        h4 = self.op32(h4)
        h5 = self.op40(h5)
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
        h1 = h1+h3
        h2 = h2+h4
        h6 = x1+h5
        h7 = self.op41(h1)
        h7 = self.op42(h7)
        h7 = self.op43(h7)
        h7 = self.op44(h7)
        if self.training:
            h7=drop_path(h7,0.2)
        h8 = x1+h7
        h9 = torch.concat([h1,h2,h6,h8], dim=1)
        h9 = self.op45(h9)
        h9 = self.op46(h9)
        h9 = self.op47(h9)
        return h9
