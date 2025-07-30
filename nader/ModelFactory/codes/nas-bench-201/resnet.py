
import torch
import torch.nn as nn


from ModelFactory.register import Registers

class BasicBlock(nn.Module):


    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )


    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + x)

class DownsampleBlock(nn.Module):


    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )


    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

@Registers.model
class resnet(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()
        block = BasicBlock
        downsample = DownsampleBlock
        dim = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True))
        layers = []
        for _ in range(5):
            layers.append(block(dim, dim))
        layers.append(downsample(dim,dim*2))
        for _ in range(5):
            layers.append(block(dim*2, dim*2))
        layers.append(downsample(dim*2,dim*4))
        for _ in range(5):
            layers.append(block(dim*4, dim*4))
        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim*4, num_classes)

    def forward(self, x):
        h = self.conv1(x)
        h = self.layers(h)
        h = self.avg_pool(h)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        return h