
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


##### VGG

def make_layers(cfg, in_channels=3, batch_norm=False, pool='avg', bias=True):
    layers = []
    for v in cfg:
        if v == 'P':
            if pool == 'avg':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=bias)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CifarVGG(nn.Module):
    def __init__(self, config, n_classes=10, pool='max'):
        super(CifarVGG, self).__init__()
        self.features = make_layers(config, pool=pool)
        self.classifier = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, n_classes))

        # preprocessing
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.2023, 0.1994, 0.2010])

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = (x - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _apply(self, fn):
        super(CifarVGG, self)._apply(fn)
        self.mean.data = fn(self.mean.data)
        self.std.data = fn(self.std.data)


class MNISTVGGStable(nn.Module):
    def __init__(self, config, n_classes=10, activ='relu'):
        super(MNISTVGGStable, self).__init__()
        self.features = make_layers(config, in_channels=1, pool='avg', bias=False, activ=activ)
        self.classifier = nn.Linear(512, n_classes, bias=False)

        # preprocessing
        self.mean = torch.tensor([0.1307])
        self.std = torch.tensor([0.3081])

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = (x - self.mean.view(1, 1, 1)) / self.std.view(1, 1, 1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _apply(self, fn):
        super(MNISTVGGStable, self)._apply(fn)
        self.mean.data = fn(self.mean.data)
        self.std.data = fn(self.std.data)


def cifar_vgg11():
    config = [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P']
    return CifarVGG(config, n_classes=10, pool='max')


def cifar_vgg16():
    config = [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P']
    return CifarVGG(config, n_classes=10, pool='max')


def mnist_vgg5_stable(activ='relu'):
    config = [64, 'P', 128, 'P', 256, 'P', 512, 'P']
    return MNISTVGGStable(config, n_classes=10, activ=activ)


##### ResNet (with no batchnorm)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_bn=False):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if use_bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if use_bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        if self.use_bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CifarResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_bn=False):
        super(CifarResNet, self).__init__()
        self.in_planes = 64
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        # preprocessing
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.2023, 0.1994, 0.2010])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_bn=self.use_bn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = (x - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
        if self.use_bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _apply(self, fn):
        super(CifarResNet, self)._apply(fn)
        self.mean.data = fn(self.mean.data)
        self.std.data = fn(self.std.data)


def cifar_resnet18():
    return CifarResNet(BasicBlock, [2,2,2,2], use_bn=False)

