import torch.nn as nn
from ModelFactory.register import Registers

@Registers.model
class nasbench201_cifar100_optimal_forRes50(nn.Module):

    def __init__(self,num_classes=100) -> None:
        super().__init__()
        block = Registers.block['nasbench201_cifar100_optimal_forRes50_base']
        stem = Registers.block['nasbench201_cifar100_optimal_forRes50_stem']
        downsample = Registers.block['nasbench201_cifar100_optimal_forRes50_downsample']
        self.stem = stem(in_channels=3,out_channels=100)
        layers = []
        for _ in range(3):
            layers.append(block(in_channels=100,out_channels=100))
        self.layer1 = nn.Sequential(*layers)
        self.downsample1 = downsample(in_channels=100,out_channels=200)
        layers = []
        for _ in range(3):
            layers.append(block(in_channels=200,out_channels=200))
        self.layer2 = nn.Sequential(*layers)
        self.downsample2 = downsample(in_channels=200,out_channels=400)
        layers = []
        for _ in range(9):
            layers.append(block(in_channels=400,out_channels=400))
        self.layer3 = nn.Sequential(*layers)
        self.downsample3 = downsample(in_channels=400,out_channels=800)
        layers = []
        for _ in range(3):
            layers.append(block(in_channels=800,out_channels=800))
        self.layer4 = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(800,num_classes)

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

