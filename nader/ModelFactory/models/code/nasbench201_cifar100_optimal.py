import torch.nn as nn
from ModelFactory.register import Registers

@Registers.model
class nasbench201_cifar100_optimal(nn.Module):

    def __init__(self,num_classes=100) -> None:
        super().__init__()
        block = Registers.block['nasbench201_cifar100_optimal_base']
        stem = Registers.block['nasbench201_cifar100_optimal_stem']
        downsample = Registers.block['nasbench201_cifar100_optimal_downsample']
        self.stem = stem(in_channels=3,out_channels=16)
        layers = []
        for _ in range(5):
            layers.append(block(in_channels=16,out_channels=16))
        self.layer1 = nn.Sequential(*layers)
        self.downsample1 = downsample(in_channels=16,out_channels=32)
        layers = []
        for _ in range(5):
            layers.append(block(in_channels=32,out_channels=32))
        self.layer2 = nn.Sequential(*layers)
        self.downsample2 = downsample(in_channels=32,out_channels=64)
        layers = []
        for _ in range(5):
            layers.append(block(in_channels=64,out_channels=64))
        self.layer3 = nn.Sequential(*layers)
        layers = []
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64,num_classes)

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

