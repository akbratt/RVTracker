import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models.densenet as densenets

def densenet121(pretrained = True):
    base_model = densenets.densenet121(pretrained=pretrained)
    return DenseNet121(base_model)

class DenseNet121(nn.Module):
    def __init__(self, model):

        super(DenseNet121, self).__init__()

        self.features = nn.Sequential(*list(model.features.children())[0:3])
        self.pool = list(model.features.children())[3]

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


        self.block0 = list(model.features.children())[4]
        self.trans0 = list(model.features.children())[5]

        self.block1 = list(model.features.children())[6]
        self.trans1 = list(model.features.children())[7]


        self.block2 = list(model.features.children())[8]
        self.trans2 = list(model.features.children())[9]


        self.block3 = list(model.features.children())[10]

        # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.up0 = nn.Upsample(scale_factor=2)
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=8)
        self.up3 = nn.Upsample(scale_factor=16)
        self.up4 = nn.Upsample(scale_factor=32)
        growth_rate=32
        bn_size=4
        drop_rate=0

        self.final_dense = densenets._DenseBlock(num_layers = 6,
                num_input_features=81, bn_size=bn_size,\
                        growth_rate=growth_rate, drop_rate=drop_rate)
        self.final = nn.Conv2d(273, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x0):
        x = torch.cat([x0,x0,x0], dim=1)
        x = self.features(x)
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
