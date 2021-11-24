import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import copy


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, cat_ids=[0,1,2,3,4,5], zero_init_residual=False,
                 fork_layer34=False, fork_layer4=False, pretrained=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        if pretrained:
            if layers == [3, 8, 36, 3]:
                key = 'resnet152'
            elif layers == [3, 4, 23, 3]:
                key = 'resnet101'
            elif layers == [3, 4, 6, 3]:
                key = 'resnet50'
            elif layers == [3, 4, 6, 3]:
                key = 'resnet34'
            elif layers == [2, 2, 2, 2]:
                key = 'resnet18'
            self.load_state_dict(model_zoo.load_url(model_urls[key]))

        self.cat_ids = cat_ids
        self.fork_layer34 = fork_layer34  # fork layer 3 & 4
        self.fork_layer4 = fork_layer4    # fork layer 4
        self.conv_lmark = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if fork_layer34:
            print(f"resnet deepcopy: layer3_13, layer3_24")
            self.layer3_13 = copy.deepcopy(self.layer3)
            self.layer3_24 = copy.deepcopy(self.layer3)
            # self.layer3_3 = copy.deepcopy(self.layer3)
            # self.layer3_4 = copy.deepcopy(self.layer3)
            # self.layer3_5 = copy.deepcopy(self.layer3)

        if fork_layer34 or fork_layer4:
            print(f"resnet deepcopy: layer4_1, layer4_2, layer4_3, layer4_4, layer4_5")
            self.layer4_1 = copy.deepcopy(self.layer4)
            self.layer4_2 = copy.deepcopy(self.layer4)
            self.layer4_3 = copy.deepcopy(self.layer4)
            self.layer4_4 = copy.deepcopy(self.layer4)
            self.layer4_5 = copy.deepcopy(self.layer4)

        # msize = 128 # mediate layer size
        # self.fc0 = nn.Sequential(nn.Linear(512 * block.expansion, msize), nn.Linear(msize, 7))
        # self.fc1 = nn.Sequential(nn.Linear(512 * block.expansion, msize), nn.Linear(msize, 3))
        # self.fc2 = nn.Sequential(nn.Linear(512 * block.expansion, msize), nn.Linear(msize, 3))
        # self.fc3 = nn.Sequential(nn.Linear(512 * block.expansion, msize), nn.Linear(msize, 4))
        # self.fc4 = nn.Sequential(nn.Linear(512 * block.expansion, msize), nn.Linear(msize, 6))
        # self.fc5 = nn.Sequential(nn.Linear(512 * block.expansion, msize), nn.Linear(msize, 3))
        self.fc0 = nn.Sequential(nn.Linear(512 * block.expansion, 7))
        self.fc1 = nn.Sequential(nn.Linear(512 * block.expansion, 3))
        self.fc2 = nn.Sequential(nn.Linear(512 * block.expansion, 3))
        self.fc3 = nn.Sequential(nn.Linear(512 * block.expansion, 4))
        self.fc4 = nn.Sequential(nn.Linear(512 * block.expansion, 6))
        self.fc5 = nn.Sequential(nn.Linear(512 * block.expansion, 3))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        if self.fork_layer34:
            x0, x1, x2, x3, x4, x5 = self._fork_layer3(x)
        else:
            x = self.layer3(x)
            x0, x1, x2, x3, x4, x5 = x, x, x, x, x, x

        if self.fork_layer34 or self.fork_layer4:
            x0, x1, x2, x3, x4, x5 = self._fork_layer4(x0, x1, x2, x3, x4, x5)
        else:
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x0, x1, x2, x3, x4, x5 = x, x, x, x, x, x

        x0 = self.fc0(x0) if 0 in self.cat_ids else None
        x1 = self.fc1(x1) if 1 in self.cat_ids else None
        x2 = self.fc2(x2) if 2 in self.cat_ids else None
        x3 = self.fc3(x3) if 3 in self.cat_ids else None
        x4 = self.fc4(x4) if 4 in self.cat_ids else None
        x5 = self.fc5(x5) if 5 in self.cat_ids else None

        return x0, x1, x2, x3, x4, x5

    def _fork_layer3(self, x):
        x05 = self.layer3(x)
        x13 = self.layer3_13(x)
        x24 = self.layer3_24(x)
        # x3 = self.layer3_3(x)
        # x4 = self.layer3_4(x)
        # x5 = self.layer3_5(x)
        return x05, x13, x24, x13, x24, x05

    def _fork_layer4(self, x0, x1, x2, x3, x4, x5):
        x0 = self.layer4(x0)
        x1 = self.layer4_1(x1)
        x2 = self.layer4_2(x2)
        x3 = self.layer4_3(x3)
        x4 = self.layer4_4(x4)
        x5 = self.layer4_5(x5)

        x0 = self.avgpool(x0)
        x1 = self.avgpool(x1)
        x2 = self.avgpool(x2)
        x3 = self.avgpool(x3)
        x4 = self.avgpool(x4)
        x5 = self.avgpool(x5)
        x0 = x0.view(x0.size(0), -1)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)
        x5 = x5.view(x5.size(0), -1)
        return x0, x1, x2, x3, x4, x5


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], pretrained=pretrained, **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], pretrained=pretrained, **kwargs)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], pretrained=pretrained, **kwargs)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], pretrained=pretrained, **kwargs)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], pretrained=pretrained, **kwargs)
    return model
