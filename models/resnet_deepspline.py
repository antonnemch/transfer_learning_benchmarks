import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import copy

from models.ds_modules import dsnn

DEFAULT_OPT_PARAMS = {
    'size': 51,
    'range_': 4,
    'init': 'leaky_relu',
    'save_memory': False
}

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, opt_params=None):
        super().__init__()
        self.opt_params = opt_params or DEFAULT_OPT_PARAMS

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = dsnn.DeepBSpline('conv', planes, **self.opt_params)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = dsnn.DeepBSpline('conv', planes, **self.opt_params)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.act3 = dsnn.DeepBSpline('conv', planes * self.expansion, **self.opt_params)

    def forward(self, x):
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)

        return out

class ResNet(dsnn.DSModule):
    def __init__(self, block, layers, num_classes=1000, opt_params=None):
        super().__init__()
        self.opt_params = opt_params or DEFAULT_OPT_PARAMS
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = dsnn.DeepBSpline('conv', 64, **self.opt_params)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.initialization(spline_init=self.opt_params['init'], init_type='He')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, opt_params=self.opt_params)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, opt_params=self.opt_params))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def freeze_all_but_splines(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.parameters_deepspline():
            param.requires_grad = True

def resnet50_deepspline(pretrained=False, num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        state_dict.pop("fc.weight", None)
        state_dict.pop("fc.bias", None)
        model.load_state_dict(state_dict, strict=False)
    return model

def initialize_spline_model(num_classes, device, freeze=True):
    model = resnet50_deepspline(pretrained=True, num_classes=num_classes)
    if freeze:
        model.freeze_all_but_splines()
    model.to(device)
    return model
