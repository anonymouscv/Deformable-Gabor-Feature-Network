import math
import torch
import torch.nn as nn
import ipdb
import sys
sys.path.append("..")
#from gcn.layers import GConv
#from dcn.layers import DConv
from dgfn.layers import DGConv
def conv3x3(in_planes, out_planes, stride=1, M=1, method='Conv', expand=False, nScale=1):
    """3x3 convolution with padding"""
    if method == 'Conv':
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    elif method == 'GConv':
        return GConv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, M=M, expand=expand, nScale=nScale)
    elif method == 'DConv':
        return DConv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    elif method == 'DGConv':
        return DGConv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, M=M, expand=expand, nScale=nScale)

    else:
        raise ValueError

def conv1x1(in_planes, out_planes, stride=1, M=1, method='Conv', expand=False, nScale=1):
    """1x1 convolution without padding"""
    if method == 'Conv' or method == 'DConv':
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    elif method == 'GConv' or method == 'DGConv':
        return GConv(in_planes, out_planes, kernel_size=1, stride=stride, M=M, bias=False, expand=expand, nScale=nScale)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, M=1, method='Conv',expand=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, M=M, method=method, expand = expand)
        self.bn1 = nn.BatchNorm2d(planes * M)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, M=M, method=method)
        self.bn2 = nn.BatchNorm2d(planes * M)
        self.downsample = downsample
        self.stride = stride
        self.method = method
        if method == 'DConv' or method == 'DGConv':
            if expand == False:
                offset_in = inplanes*M
            else:
                offset_in = inplanes
            self.off_conv1 = nn.Conv2d(offset_in, 18, kernel_size=3, stride=stride, padding=1, bias=False)
            self.off_conv2 = nn.Conv2d(planes*M, 18, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        residual = x
        if self.method == 'DConv' or self.method == 'DGConv':
            off1 = self.off_conv1(x)
            out = self.conv1(x,off1)
        else:
            out = self.conv1(x)
        #ipdb.set_trace()    
        out = self.bn1(out)
        out = self.relu(out)
        if self.method == 'DConv' or self.method == 'DGConv':
            off2 = self.off_conv2(out)
            out = self.conv2(out,off2)
        else:
            out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        #print(out.size(),residual.size())
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, stages=[8,16,32,64], num_classes=10, M=1, stage_with_method=[True, True, True, True], method='Conv'):
        methods = [method, method, method, method]
        expands = [False, False, False, False]
        for i in range(4):
            if stage_with_method[i] == False:
                methods[i] = 'Conv'
        self.method = method
        if method == 'GConv' or method == 'DGConv':
            if stage_with_method[0] == True:
                expands[0] = True
            else:    
                for i in range(3):
                    if stage_with_method[i] == False and stage_with_method[i+1] == True:
                        expands[i+1] = True
                        break
                 
     
        self.inplanes = stages[0]
        super(ResNet, self).__init__()
        self.M = M
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv1 = conv3x3(3, self.inplanes, stride=1, M=M, method=method_back, expand=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes )
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, stages[0], layers[0], M=M, method=methods[0], expand = expands[0])
        self.layer2 = self._make_layer(block, stages[1], layers[1], stride=2, M=M, method=methods[1], expand = expands[1])
        self.layer3 = self._make_layer(block, stages[2], layers[2], stride=2, M=M, method=methods[2], expand = expands[2])
        self.layer4 = self._make_layer(block, stages[3], layers[3], stride=2, M=M, method=methods[3], expand = expands[3])
        #self.avgpool = nn.AvgPool2d(4, stride=1)
        self.adapool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stages[3] * block.expansion * M, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1, M=1, method='Conv', expand = False):
        downsample = None
        if method == 'Conv' or method == 'DConv':
            M=1
        self.M = M
        if stride != 1 or self.inplanes != planes * block.expansion:
            if expand == False:
                down_in = self.inplanes*M
            else:
                down_in = self.inplanes
            downsample = nn.Sequential(
                conv1x1(down_in, planes*block.expansion*self.M, stride=stride, M=M, nScale=1),
                nn.BatchNorm2d(planes * block.expansion * M),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, M=M, method=method, expand = expand))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, M=M, method=method))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adapool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, M=1, method='Conv', set_method=[False, True, True, True],stages=[64,64,128,256]):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print(stages)
    assert method in ['Conv', 'GConv', 'DConv', 'DGConv']
    if method=='Conv' or method=='DConv':
        M = 1
    model = ResNet(BasicBlock, [2, 2, 2, 2], M=M, method=method, stage_with_method=set_method, stages=stages)
    if pretrained:
        raise ValueError
    return model
