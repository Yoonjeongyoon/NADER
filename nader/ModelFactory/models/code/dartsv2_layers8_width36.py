import torch.nn as nn
from ModelFactory.register import Registers

class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
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
class dartsv2_layers8_width36(nn.Module):

    def __init__(self,num_classes=100,auxiliary=False) -> None:
        super().__init__()
        self.auxiliary = auxiliary
        base = Registers.block['dartsv2_layers8_width36_base']
        stem = Registers.block['dartsv2_layers8_width36_stem']
        downsample = Registers.block['dartsv2_layers8_width36_downsample']
        self.stem = stem(3,36)
        cells_type = [0]*2+[1]+[0]*2+[1]+[0]*2
        cell_widths = [36]*2+[72]+[72]*2+[144]+[144]*2
        self.layer_num = sum([2, 2, 2])+2
        self.cells = nn.ModuleList()
        reduction_pre,reduction_pre_pre = False, False
        width_pre,width_pre_pre = 36,36
        for i in range(self.layer_num):
            if i in [self.layer_num//3,2*self.layer_num//3]:
                cell = downsample(width_pre,cell_widths[i],width_pre_pre,reduction_pre_pre)
                reduction_pre = True
            else:
                cell = base(width_pre,cell_widths[i],width_pre_pre,reduction_pre_pre)
                reduction_pre = False
            width_pre_pre = width_pre
            width_pre = cell_widths[i]
            self.cells += [cell]
            reduction_pre_pre = reduction_pre
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(144, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(144,num_classes)

    def forward(self,x):
        logits_aux = None
        x1 = x2 = self.stem(x)
        for i,cell in enumerate(self.cells):
            x1,x2 = x2,cell(x1,x2)
            if i == 2*self.layer_num//3:
                if self.auxiliary and self.training:
                    logits_aux = self.auxiliary_head(x2)
        x2 = self.avg_pool(x2)
        x2 = x2.view(x2.size(0), -1)
        logits = self.fc(x2)
        return logits,logits_aux
        
