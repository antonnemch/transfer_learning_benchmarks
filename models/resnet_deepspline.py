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

    def __init__(self, in_planes, planes, stride=1, downsample=None,
                 opt_params=None, shared_act=None):
        super(Bottleneck, self).__init__()
        self.opt_params = opt_params
        self.expansion = Bottleneck.expansion
        self.downsample = downsample
        self.stride = stride

        # Convolution and BatchNorm layers (needed regardless of sharing)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        # Activation sharing logic
        sharing = self.opt_params.get("sharing", "channel")

        if isinstance(shared_act, dict):  # model-level with separate act3
            self.act1 = shared_act['act']
            self.act2 = shared_act['act']
            self.act3 = shared_act['act3']
        elif shared_act is not None:
            self.act1 = shared_act
            self.act2 = shared_act
            self.act3 = shared_act
        elif sharing == "block":
            self.shared_act = dsnn.DeepBSpline('conv', planes, **self.opt_params)
            self.act1 = self.act2 = self.shared_act
            self.act3 = dsnn.DeepBSpline('conv', planes * self.expansion, **self.opt_params)
        else:
            self.act1 = dsnn.DeepBSpline('conv', planes, **self.opt_params)
            self.act2 = dsnn.DeepBSpline('conv', planes, **self.opt_params)
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
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = dsnn.DeepBSpline('conv', 64, **self.opt_params)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.sharing = self.opt_params.get("sharing", "channel")
        self.global_act = None
        if self.sharing == "model":
            self.global_acts = {}  # key: planes value: DeepBSpline
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.initialization(spline_init=self.opt_params['init'], init_type='He')

    def _make_layer(self, block, planes, blocks, stride=1):
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        else:
            downsample = None

        layers = []

        # 👇 Determine shared activation based on sharing level
        shared_layer_act = None
        if self.sharing == "model":
            if planes not in self.global_acts:
                self.global_acts[planes] = dsnn.DeepBSpline('conv', planes, **self.opt_params)
            if planes * block.expansion not in self.global_acts:
                self.global_acts[planes * block.expansion] = dsnn.DeepBSpline('conv', planes * block.expansion, **self.opt_params)
            
            # Pass both activations into the block
            shared_layer_act = {
                'act': self.global_acts[planes],
                'act3': self.global_acts[planes * block.expansion]
            }
        elif self.sharing == "layer":
            shared_layer_act = {
                'act': dsnn.DeepBSpline('conv', planes, **self.opt_params),
                'act3': dsnn.DeepBSpline('conv', planes * block.expansion, **self.opt_params)
            }

        # 👇 First block (with optional downsample)
        layers.append(block(
            self.in_planes,
            planes,
            stride,
            downsample,
            opt_params=self.opt_params,
            shared_act=shared_layer_act
        ))

        self.in_planes = planes * block.expansion

        # 👇 Remaining blocks in this layer
        for _ in range(1, blocks):
            layers.append(block(
                self.in_planes,
                planes,
                opt_params=self.opt_params,
                shared_act=shared_layer_act
            ))

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

def initialize_spline_model(num_classes, device, freeze=True,sharing='channel'):
    DEFAULT_OPT_PARAMS['sharing'] = sharing  # 'channel', 'layer', 'block', 'model'
    model = resnet50_deepspline(pretrained=True, num_classes=num_classes)
    if freeze:
        model.freeze_all_but_splines()
    model.to(device)
    return model
