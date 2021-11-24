import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, model_name=None):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )
        if init_weights:
            self._initialize_weights()
        else:
            self.load_state_dict(model_zoo.load_url(model_urls[model_name]))
        self.classifier = self.classifier[:-1]
        self.fc0 = nn.Linear(4096, 7)
        self.fc1 = nn.Linear(4096, 3)
        self.fc2 = nn.Linear(4096, 3)
        self.fc3 = nn.Linear(4096, 4)
        self.fc4 = nn.Linear(4096, 6)
        self.fc5 = nn.Linear(4096, 3)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x0 = self.fc0(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        x5 = self.fc5(x)
        return x0, x1, x2, x3, x4, x5

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model2 (configuration "A")

    Args:
        pretrained (bool): If True, returns a model2 pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    kwargs['model_name'] = 'vgg11'
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model2 (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model2 pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    kwargs['model_name'] = 'vgg11_bn'
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model2 (configuration "B")

    Args:
        pretrained (bool): If True, returns a model2 pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    kwargs['model_name'] = 'vgg13'
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model2 (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model2 pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    kwargs['model_name'] = 'vgg13_bn'
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model2 (configuration "D")

    Args:
        pretrained (bool): If True, returns a model2 pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    kwargs['model_name'] = 'vgg16'
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model2 (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model2 pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    kwargs['model_name'] = 'vgg16_bn'
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model2 (configuration "E")

    Args:
        pretrained (bool): If True, returns a model2 pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    kwargs['model_name'] = 'vgg19'
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model2 (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model2 pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    kwargs['model_name'] = 'vgg19_bn'
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model
