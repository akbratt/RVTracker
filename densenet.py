import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet121']))
    return model

def densenet_small(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(4, 4,
        4, 4),
                     **kwargs)
    # Have tried (6,6,6,6)[overfit], (2,4,6,6)[probably underfit], (4,4,6,6)
    # [overfit]
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet121']))
    return model

def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet169']))
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet201']))
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet161']))
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Each denseblock
        # num_features = num_init_features
        # for i, num_layers in enumerate(block_config):
            # block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                # bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            # self.features.add_module('denseblock%d' % (i + 1), block)
            # num_features = num_features + num_layers * growth_rate
            # if i != len(block_config) - 1:
                # trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                # self.features.add_module('transition%d' % (i + 1), trans)
                # num_features = num_features // 2


        num_features = num_init_features
        self.block0 = _DenseBlock(num_layers=block_config[0],
                num_input_features=num_features, bn_size=bn_size, \
                        growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_config[0] * growth_rate
        self.trans0 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2


        self.block1 = _DenseBlock(num_layers=block_config[1],
                num_input_features=num_features, bn_size=bn_size, \
                        growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_config[1] * growth_rate
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2


        self.block2 = _DenseBlock(num_layers=block_config[2],
                num_input_features=num_features, bn_size=bn_size, \
                        growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_config[2] * growth_rate
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2


        self.block3 = _DenseBlock(num_layers=block_config[3],
                num_input_features=num_features, bn_size=bn_size, \
                        growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_config[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_features)

        # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        self.up0 = nn.Upsample(scale_factor=2)
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=8)
        self.up3 = nn.Upsample(scale_factor=16)
        self.up4 = nn.Upsample(scale_factor=32)

        self.final_dense = _DenseBlock(num_layers = 6,
                num_input_features=81, bn_size=bn_size,\
                        growth_rate=growth_rate, drop_rate=drop_rate)
        self.final = nn.Conv2d(273, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x0):
        x = self.features(x0)
        cat0 = self.up0(x)[:, -16:]
        x = self.pool(x)

        x = self.block0(x)
        cat1 = self.up1(x)[:, -16:]
        x = self.trans0(x)

        x = self.block1(x)
        cat2 = self.up2(x)[:, -16:]
        x = self.trans1(x)

        x = self.block2(x)
        cat3 = self.up3(x)[:, -16:]
        x = self.trans2(x)

        x = self.block3(x)
        cat4 = self.up4(x)[:, -16:]

        out = torch.cat([x0, cat0, cat1, cat2, cat3, cat4], dim=1)
        out = self.final_dense(out)

        out = self.final(out)

        return out
