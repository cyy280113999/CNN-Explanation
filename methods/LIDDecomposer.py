"""
LID Decomposer for both linear and nonlinear.

compose linear & nonlinear in one.

use linear=False parameter.

"""

import torch
import torch.nn.functional as nf
from torchvision.models import VGG, AlexNet, ResNet, GoogLeNet, VisionTransformer
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.googlenet import BasicConv2d, Inception

from utils import *

ABBREV = False

# kwargs: x0="std0", BP="normal", LIN=0, DEFAULT_STEP=11, LAE=0, CAE=0, AMP=0, DF=0, GIP=0
def LID_wrapper(model, layer_names, **kwargs):  # class-like
    method = LIDDecomposer(model, **kwargs)
    def data_caller(x, y):  # a heatmap method
        method(x, y)
        hm = multi_interpolate(relevanceFindByName(model, layer_name) for layer_name in layer_names)
        return hm
    return data_caller


# this gives the stage wrapper for common nets.
# and multi-layer mixed heatmap
def LID_m_caller(model, x, y, x0='std0',
                 s=(0, 1, 2, 3, 4, 5), lin=False, bp='sig', le=0, ce=0, smg=0):
    d = LIDDecomposer(model, x0=x0, BP=bp, LIN=lin, LAE=le, CAE=ce, GIP=smg)
    d(x, y)
    hm = multi_interpolate(relevanceFindByName(model, layername)
                           for layername in decode_stages(model, s))
    return hm


"""
LRP-0 can easily got by LID with zero layer reference.
These pairs are same:
    "LRP-0-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='normal', method='lrp0', layer_num=-1)),
    "LID-LRP-0-f": lambda model: lambda x, y: LRP_caller(model, x, y, layer_name=('features', -1), bp=None),
    "LRP-0-24": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='normal', method='lrp0', layer_num=24)),
    "LID-LRP-0-23": lambda model: lambda x, y: LRP_caller(model, x, y, layer_name=('features', 23), bp=None),
ST-LRP=ST-LID when layer refer to zero
    "ST-LRP-0-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='st', method='lrp0', layer_num=-1)),
    "ST-LID-LRP-0-f": lambda model: lambda x, y: LRP_caller(model, x, y, layer_name=('features', -1), bp='st'),
    "ST-LRP-0-24": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='st', method='lrp0', layer_num=24)),
    "ST-LID-LRP-0-23": lambda model: lambda x, y: LRP_caller(model, x, y, layer_name=('features', 23), bp='st'),
and this is a little different. LRP refer to 0, and LID refer to a refer-logits
    "SIG-LRP-0-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='sig', method='lrp0', layer_num=-1)),
    "SIG-LID-LRP-0-f": lambda model: lambda x, y: LRP_caller(model, x, y, layer_name=('features', -1), bp='sig'),
    "SIG-LRP-0-24": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='sig', method='lrp0', layer_num=24)),
    "SIG-LID-LRP-0-23": lambda model: lambda x, y: LRP_caller(model, x, y, layer_name=('features', 23), bp='sig'),

"""


def LRP_caller(model, x, y, x0='std0', layer_name=('features', -1), bp='sig'):
    d = LIDDecomposer(model, LIN=True)
    d(x, y, x0, bp)
    layer = findLayerByName(model, layer_name)
    r = layer.y[1] * layer.g  # this means refer to zero. A-0 = A itself.
    hm = interpolate_to_imgsize(r)
    return hm


"""
like RelevanceCAM, aims to reducing grid pattern
    "LID-CAM-ST": lambda model: lambda x, y: interpolate_to_imgsize(
        LID_CAM(model, x, y, x0='std0', layer_name=('layer1', -1, 'relu2'), bp='st', linear=False)),
    "LID-CAM-c": lambda model: lambda x, y: interpolate_to_imgsize(
        LID_CAM(model, x, y, x0='std0', layer_name=('layer1', -1, 'relu2'), bp='c', linear=False)),
    "LID-CAM-1": lambda model: lambda x, y: interpolate_to_imgsize(
        LID_CAM(model, x, y, x0='std0', layer_name=('layer1', -1, 'relu2'), bp='sig', linear=False)),
but bad heatmap when into lower layer.
    "LID-CAM-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LID_CAM(model, x, y, x0='std0', layer_name=('layer4', -1), bp='st', linear=False)),
    "LID-CAM-23": lambda model: lambda x, y: interpolate_to_imgsize(
        LID_CAM(model, x, y, x0='std0', layer_name=('layer3', -1), bp='st', linear=False)),
    "LID-CAM-16": lambda model: lambda x, y: interpolate_to_imgsize(
        LID_CAM(model, x, y, x0='std0', layer_name=('layer2', -1), bp='st', linear=False)),
    "LID-CAM-9": lambda model: lambda x, y: interpolate_to_imgsize(
        LID_CAM(model, x, y, x0='std0', layer_name=('layer1', -1), bp='st', linear=False)),
    "LID-CAM-0": lambda model: lambda x, y: interpolate_to_imgsize(
        LID_CAM(model, x, y, x0='std0', layer_name=('maxpool', ), bp='st', linear=False)),
"""


def LID_CAM(model, x, y, x0='std0', layer_name=('features', -1), bp='sig', linear=False):
    d = LIDDecomposer(model, LIN=linear, DEFAULT_STEP=11)
    d(x, y, x0, bp)
    layer = findLayerByName(model, layer_name)
    r = layer.y.diff(dim=0) * layer.g
    return (r.sum([2, 3], True) * layer.y[1]).sum(1, True)


"""
this give pixel level image
"""
# def LID_image(model, x, y, x0='std0', bp='sig', linear=False):
#     d = LIDDecomposer(model, LINEAR=linear, DEFAULT_STEP=11)
#     g, rx = d(x, y, x0, bp)
#     rx = heatmapNormalizeR(rx.detach().cpu().clip(min=0))  # how to use negative?
#     # r = d.backward(y, bp).detach().cpu()
#     # stdr=r.std(dim=[2,3], keepdim=True)
#     # r = ImgntStdTensor/stdr * r
#     # r = invStd(r)
#     return rx


# def LID_grad(model, x, y, x0='std0', bp='sig', linear=False):
#     d = LIDDecomposer(model, LINEAR=linear, DEFAULT_STEP=11)
#     g, rx = d(x, y, x0, bp)
#     g = heatmapNormalizeR(g.detach().cpu().sum(1, True))
#     return g


# take into calculation
BaseUnits = (
    # special
    torch.nn.Flatten,
    torch.nn.Dropout,
    torch.nn.ReLU,
    # linear
    torch.nn.Conv2d,
    torch.nn.BatchNorm2d,
    torch.nn.Linear,
    torch.nn.AvgPool2d,
    torch.nn.LayerNorm,
    # nonlinear
    torch.nn.MaxPool2d,
    torch.nn.AdaptiveAvgPool2d,
    torch.nn.Softmax,
)

# these modules working non-linear as same as the linear
LinearUnits = (
    torch.nn.Conv2d,
    torch.nn.BatchNorm2d,
    torch.nn.Linear,
    torch.nn.AvgPool2d,
    torch.nn.LayerNorm,
)

# some module's descriptions
SpecificUnits = (
    torch.nn.Flatten,  # if manually reshaping in forward, manually reshaping in backward
    torch.nn.Dropout,  # to ignore
    torch.nn.ReLU,  # set inplace to False
    BasicBlock,  # resnet block
    Bottleneck,  #
    Inception,  # google net block
)


def linearEnhance(layer, gamma=0.5):
    if isinstance(layer, torch.nn.Linear):
        new_layer = torch.nn.Linear(layer.in_features, layer.out_features,
                                    bias=False, device=device)
    elif isinstance(layer, torch.nn.Conv2d):
        new_layer = torch.nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size,
                                    layer.stride, layer.padding, layer.dilation,
                                    layer.groups, padding_mode=layer.padding_mode,
                                    bias=False, device=device)
    else:
        return layer
    new_layer.weight = torch.nn.Parameter(
        layer.weight +
        gamma * layer.weight.clip(min=0)
    )
    return new_layer


def downSampleFix(g):
    quarter = g[:, :, :-1, :-1]
    quarter /= 4
    quarter = quarter.clone()
    g[:, :, :-1, 1:] += quarter
    g[:, :, 1:, :-1] += quarter
    g[:, :, 1:, 1:] += quarter
    return g


class LIDDecomposer:
    def forward_baseunit(self, m, x):
        if isinstance(m, torch.nn.ReLU):
            m.inplace = False
        elif isinstance(m, torch.nn.Dropout):
            return x
        y = m(x)
        m.x = x.clone().detach()
        m.y = y.clone().detach()
        return y

    def backward_linearunit(self, m, g):
        m.g = g.clone().detach()
        x = m.x[None, 1].detach().requires_grad_()  # new graph
        if self.LinearActivationEnhance != 0 and isinstance(m, torch.nn.Linear):
            m = linearEnhance(m, self.LinearActivationEnhance)
        if self.ConvActivationEnhance != 0 and isinstance(m, torch.nn.Conv2d):
            m = linearEnhance(m, self.ConvActivationEnhance)
        with torch.enable_grad():
            y = m(x)
            (y * g).sum().backward()
        g = x.grad.clone().detach()
        return g

    def backward_nonlinearunit(self, m, g):
        m.g = g.clone().detach()
        step = self.DEFAULT_STEP
        xs = torch.zeros((step,) + m.x.shape[1:], device=m.x.device)
        xs[0] = m.x[0]
        Dx = m.x.diff(dim=0)
        dx = Dx / (step - 1)
        std = Dx.std()
        for i in range(1, step):
            xs[i] = xs[i - 1] + dx
        if self.GaussianIntegralPath:
            xs += self.GaussianIntegralPath * dx * torch.randn_like(xs)
        if self.AverageMaxPool and isinstance(m, torch.nn.MaxPool2d):
            m = torch.nn.AvgPool2d(m.kernel_size, m.stride, m.padding, ceil_mode=m.ceil_mode)
        xs = xs.detach().requires_grad_()
        with torch.enable_grad():
            ys = m(xs)
            (ys * g).sum().backward()
        g = xs.grad.mean(0, True).detach()
        return g

    def backward_baseunit(self, m, g):
        # m.Ry = m.y.diff(dim=0).detach()
        if isinstance(m, torch.nn.Flatten):
            return g.reshape((1,) + m.x.shape[1:])
        elif isinstance(m, torch.nn.Dropout):
            return g
        elif self.LinearDecomposing or isinstance(m, LinearUnits):  # nonlinear module using linear approximation
            g = self.backward_linearunit(m, g)
        else:  # nonlinear
            g = self.backward_nonlinearunit(m, g)
        return g

    def forward_vgg(self, x):
        for i, m in enumerate(self.model.features):
            x = self.forward_baseunit(m, x)
        x = self.forward_baseunit(self.model.avgpool, x)
        self.model.flatten = torch.nn.Flatten()
        x = self.forward_baseunit(self.model.flatten, x)
        for m in self.model.classifier:
            x = self.forward_baseunit(m, x)
        return x

    def backward_vgg(self, g):
        for m in self.model.classifier[::-1]:
            g = self.backward_baseunit(m, g)
        g = self.backward_baseunit(self.model.flatten, g)
        g = self.backward_baseunit(self.model.avgpool, g)
        for m in self.model.features[::-1]:
            g = self.backward_baseunit(m, g)
        return g

    # def forward_BasicBlock(self, m, x):
    #     return self.forward_baseunit(m, x)
    def forward_BasicBlock(self, m, x):
        identity = x
        x = self.forward_baseunit(m.conv1, x)
        x = self.forward_baseunit(m.bn1, x)
        x = self.forward_baseunit(m.relu, x)
        x = self.forward_baseunit(m.conv2, x)
        x = self.forward_baseunit(m.bn2, x)
        if m.downsample is not None:
            for m2 in m.downsample:
                identity = self.forward_baseunit(m2, identity)
        x += identity
        m.relu2 = torch.nn.ReLU(False)
        x = self.forward_baseunit(m.relu2, x)
        m.y = x
        return x

    # def backward_BasicBlock(self, m, g):
    #     m.g=g
    #     return self.backward_nonlinearunit(m, g)

    def backward_BasicBlock(self, m, g):
        m.g = g  # save for block relevance
        g = self.backward_baseunit(m.relu2, g)
        out_g = g
        if m.downsample is not None:
            m2=None
            for m2 in m.downsample[::-1]:
                out_g = self.backward_baseunit(m2, out_g)
            if self.ResNetDownSampleFix and m2.stride[0]!=1:
                out_g = downSampleFix(out_g)
        g = self.backward_baseunit(m.bn2, g)
        g = self.backward_baseunit(m.conv2, g)
        g = self.backward_baseunit(m.relu, g)
        g = self.backward_baseunit(m.bn1, g)
        g = self.backward_baseunit(m.conv1, g)
        g += out_g
        return g

    # def forward_Bottleneck(self, m, x, abbrev=ABBREV):
    #     return self.forward_baseunit(m, x)

    def forward_Bottleneck(self, m, x, abbrev=ABBREV):
        if abbrev:
            m.x = x
            x = m(x)
        else:
            identity = x
            x = self.forward_baseunit(m.conv1, x)
            x = self.forward_baseunit(m.bn1, x)
            x = self.forward_baseunit(m.relu, x)
            x = self.forward_baseunit(m.conv2, x)
            x = self.forward_baseunit(m.bn2, x)
            m.relu2 = torch.nn.ReLU(False)
            x = self.forward_baseunit(m.relu2, x)
            x = self.forward_baseunit(m.conv3, x)
            x = self.forward_baseunit(m.bn3, x)
            if m.downsample is not None:
                for sub_m in m.downsample:
                    identity = self.forward_baseunit(sub_m, identity)
            x += identity
            m.relu3 = torch.nn.ReLU(False)
            x = self.forward_baseunit(m.relu3, x)
        m.y = x
        return x

    # def backward_Bottleneck(self, m, g, abbrev=ABBREV):
    #     m.g=g
    #     return self.backward_nonlinearunit(m, g)

    def backward_Bottleneck(self, m, g, abbrev=ABBREV):
        m.g = g
        if abbrev:
            g = self.backward_nonlinearunit(m, g)
        else:
            g = self.backward_baseunit(m.relu3, g)
            out_g = g
            if m.downsample is not None:
                sub_m=None
                for sub_m in m.downsample[::-1]:
                    out_g = self.backward_baseunit(sub_m, out_g)
                if self.ResNetDownSampleFix and sub_m.stride[0]!=1:
                    out_g = downSampleFix(out_g)
            g = self.backward_baseunit(m.bn3, g)
            g = self.backward_baseunit(m.conv3, g)
            g = self.backward_baseunit(m.relu2, g)
            g = self.backward_baseunit(m.bn2, g)
            g = self.backward_baseunit(m.conv2, g)
            g = self.backward_baseunit(m.relu, g)
            g = self.backward_baseunit(m.bn1, g)
            g = self.backward_baseunit(m.conv1, g)
            g += out_g
        return g

    def forward_resnet(self, x):
        x = self.forward_baseunit(self.model.conv1, x)
        x = self.forward_baseunit(self.model.bn1, x)
        x = self.forward_baseunit(self.model.relu, x)
        x = self.forward_baseunit(self.model.maxpool, x)
        if isinstance(self.model.layer1[0], BasicBlock):
            self.forward_block = self.forward_BasicBlock
            self.backward_block = self.backward_BasicBlock
        else:
            self.forward_block = self.forward_Bottleneck
            self.backward_block = self.backward_Bottleneck
        for m in self.model.layer1:
            x = self.forward_block(m, x)
        for m in self.model.layer2:
            x = self.forward_block(m, x)
        for m in self.model.layer3:
            x = self.forward_block(m, x)
        for m in self.model.layer4:
            x = self.forward_block(m, x)
        x = self.forward_baseunit(self.model.avgpool, x)
        self.model.flatten = torch.nn.Flatten(1)
        x = self.forward_baseunit(self.model.flatten, x)
        x = self.forward_baseunit(self.model.fc, x)
        return x

    def backward_resnet(self, g):
        g = self.backward_baseunit(self.model.fc, g)
        g = self.backward_baseunit(self.model.flatten, g)
        g = self.backward_baseunit(self.model.avgpool, g)
        for m in self.model.layer4[::-1]:
            g = self.backward_block(m, g)
        for m in self.model.layer3[::-1]:
            g = self.backward_block(m, g)
        for m in self.model.layer2[::-1]:
            g = self.backward_block(m, g)
        for m in self.model.layer1[::-1]:
            g = self.backward_block(m, g)
        g = self.backward_baseunit(self.model.maxpool, g)
        g = self.backward_baseunit(self.model.relu, g)
        g = self.backward_baseunit(self.model.bn1, g)
        g = self.backward_baseunit(self.model.conv1, g)
        return g

    def forward_BasicConv2d(self, m, x):
        x = self.forward_baseunit(m.conv, x)
        x = self.forward_baseunit(m.bn, x)
        m.relu = torch.nn.ReLU(False)
        x = self.forward_baseunit(m.relu, x)
        return x

    def backward_BasicConv2d(self, m, g):
        g = self.backward_baseunit(m.relu, g)
        g = self.backward_baseunit(m.bn, g)
        g = self.backward_baseunit(m.conv, g)
        return g

    def forward_inception(self, m, x):
        x1 = x2 = x3 = x4 = x
        x1 = self.forward_BasicConv2d(m.branch1, x1)
        for mm in m.branch2:
            x2 = self.forward_BasicConv2d(mm, x2)
        for mm in m.branch3:
            x3 = self.forward_BasicConv2d(mm, x3)
        x4 = self.forward_baseunit(m.branch4[0], x4)
        x4 = self.forward_BasicConv2d(m.branch4[1], x4)
        x = torch.cat([x1, x2, x3, x4], 1)
        m.y = x
        return x

    def backward_inception(self, m, g):
        m.g = g
        x2_start = m.branch1.relu.y[1].shape[0]
        x2_len = m.branch2[-1].relu.y[1].shape[0]
        x3_start = x2_start + x2_len
        x3_len = m.branch3[-1].relu.y[1].shape[0]
        x4_start = x3_start + x3_len
        g1 = g[0, :x2_start].unsqueeze(0)
        g2 = g[0, x2_start:x3_start].unsqueeze(0)
        g3 = g[0, x3_start:x4_start].unsqueeze(0)
        g4 = g[0, x4_start:].unsqueeze(0)
        g1 = self.backward_BasicConv2d(m.branch1, g1)
        for mm in m.branch2[::-1]:
            g2 = self.backward_BasicConv2d(mm, g2)
        for mm in m.branch3[::-1]:
            g3 = self.backward_BasicConv2d(mm, g3)
        g4 = self.backward_BasicConv2d(m.branch4[1], g4)
        g4 = self.backward_baseunit(m.branch4[0], g4)
        g = g1 + g2 + g3 + g4
        return g

    def forward_googlenet(self, x):
        # x = invStd(x) * 2 - 1  # this with some small error/gap.
        self.model.t_m = torch.tensor([(0.229 / 0.5), (0.224 / 0.5), (0.225 / 0.5)],
                                      device=x.device).reshape(1, -1, 1, 1)
        self.model.t_b = torch.tensor([(0.485 - 0.5) / 0.5, (0.456 - 0.5) / 0.5, (0.406 - 0.5) / 0.5],
                                      device=x.device).reshape(1, -1, 1, 1)
        # googlenet transform modified by me.
        # this with same output but grad differs from googlenet ? Due to graph?
        x = x * self.model.t_m + self.model.t_b
        x = self.forward_BasicConv2d(self.model.conv1, x)  # nonlinear
        x = self.forward_baseunit(self.model.maxpool1, x)
        x = self.forward_BasicConv2d(self.model.conv2, x)
        x = self.forward_BasicConv2d(self.model.conv3, x)
        x = self.forward_baseunit(self.model.maxpool2, x)
        x = self.forward_inception(self.model.inception3a, x)
        x = self.forward_inception(self.model.inception3b, x)
        x = self.forward_baseunit(self.model.maxpool3, x)
        x = self.forward_inception(self.model.inception4a, x)
        x = self.forward_inception(self.model.inception4b, x)
        x = self.forward_inception(self.model.inception4c, x)
        x = self.forward_inception(self.model.inception4d, x)
        x = self.forward_inception(self.model.inception4e, x)
        x = self.forward_baseunit(self.model.maxpool4, x)
        x = self.forward_inception(self.model.inception5a, x)
        x = self.forward_inception(self.model.inception5b, x)
        x = self.forward_baseunit(self.model.avgpool, x)
        self.model.flatten = torch.nn.Flatten(1)
        x = self.forward_baseunit(self.model.flatten, x)
        x = self.forward_baseunit(self.model.fc, x)
        return x

    def backward_googlenet(self, g):
        g = self.backward_baseunit(self.model.fc, g)
        g = self.backward_baseunit(self.model.flatten, g)
        g = self.backward_baseunit(self.model.avgpool, g)
        g = self.backward_inception(self.model.inception5b, g)
        g = self.backward_inception(self.model.inception5a, g)
        g = self.backward_baseunit(self.model.maxpool4, g)
        g = self.backward_inception(self.model.inception4e, g)
        g = self.backward_inception(self.model.inception4d, g)
        g = self.backward_inception(self.model.inception4c, g)
        g = self.backward_inception(self.model.inception4b, g)
        g = self.backward_inception(self.model.inception4a, g)
        g = self.backward_baseunit(self.model.maxpool3, g)
        g = self.backward_inception(self.model.inception3b, g)
        g = self.backward_inception(self.model.inception3a, g)
        g = self.backward_baseunit(self.model.maxpool2, g)
        g = self.backward_BasicConv2d(self.model.conv3, g)
        g = self.backward_BasicConv2d(self.model.conv2, g)
        g = self.backward_baseunit(self.model.maxpool1, g)
        g = self.backward_BasicConv2d(self.model.conv1, g)
        g = g * self.model.t_m
        return g

    class VITSA(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def __call__(self, x):
            return self.m(x, x, x, need_weights=False)[0]

    def forward_encoder_block(self, m, x):
        tmp = x
        x = self.forward_baseunit(m.ln_1, x)
        m.SA = self.VITSA(m.self_attention)
        x = self.forward_baseunit(m.SA, x)
        x = x + tmp
        tmp = x
        x = self.forward_baseunit(m.ln_2, x)
        x = self.forward_baseunit(m.mlp, x)
        x = x + tmp
        m.y = x[:, 1:].clone().permute(0, 2, 1).reshape(2, -1, 14, 14)  # modified as heatmap
        return x

    def backward_encoder_block(self, m, g):
        m.g = g[:, 1:].clone().permute(0, 2, 1).reshape(1, -1, 14, 14)  # modified as heatmap
        out_g = g
        if self.LinearDecomposing:
            g = self.backward_linearunit(m.mlp, g)
        else:
            g = self.backward_nonlinearunit(m.mlp, g)  # explicitly calling
        g = self.backward_linearunit(m.ln_2, g)
        g = g + out_g
        out_g = g
        if self.LinearDecomposing:
            g = self.backward_linearunit(m.SA, g)
        else:
            g = self.backward_nonlinearunit(m.SA, g)
        g = self.backward_linearunit(m.ln_1, g)
        g = g + out_g
        return g

    def forward_vit(self, x):
        assert isinstance(self.model, VisionTransformer)
        n, c, h, w = x.shape
        assert h == 224 and w == 224
        assert self.model.patch_size == 16
        # p = self.model.patch_size  # 16
        # n_h = h // p  # 14
        # n_w = w // p
        x = self.forward_baseunit(self.model.conv_proj, x)
        x = x.reshape(n, self.model.hidden_dim, 196)  # n,hid,14**2
        x = x.permute(0, 2, 1)  # n,196,hid
        batch_class_token = self.model.class_token.expand(x.shape[0], -1, -1)  # n,1,hid
        x = torch.cat([batch_class_token, x], dim=1)  # n,197,hid
        x = x + self.model.encoder.pos_embedding
        for m in self.model.encoder.layers:
            x = self.forward_encoder_block(m, x)
        x = self.forward_baseunit(self.model.encoder.ln, x)
        x = x[:, 0]
        for m in self.model.heads:
            x = self.forward_baseunit(m, x)
        return x

    def backward_vit(self, g):
        for m in self.model.heads[::-1]:
            g = self.backward_baseunit(m, g)
        tmp = torch.zeros_like(self.model.encoder.ln.y[1, None])
        tmp[:, 0] = g
        g = tmp
        g = self.backward_baseunit(self.model.encoder.ln, g)
        for m in self.model.encoder.layers[::-1]:
            g = self.backward_encoder_block(m, g)
        g = g[:, 1:]
        g = g.permute(0, 2, 1)
        g = g.reshape(-1, self.model.hidden_dim, 14, 14)
        g = self.backward_baseunit(self.model.conv_proj, g)
        return g

    def __init__(self, model, x0="std0", BP="normal", LIN=0, DEFAULT_STEP=21, LAE=0, CAE=0, AMP=0, DF=0, GIP=1.0, **kwargs):
        self.x0=x0
        self.backward_init=BP
        self.LinearDecomposing = LIN  # set to nonlinear decomposition
        self.DEFAULT_STEP = DEFAULT_STEP  # step of nonlinear integral approximation
        self.GaussianIntegralPath = GIP
        self.LinearActivationEnhance = LAE  # positive enhancement
        self.ConvActivationEnhance = CAE
        self.AverageMaxPool = AMP  # bad effect
        self.ResNetDownSampleFix = DF  # good for resnet. remove grid-effect-#
        if isinstance(model, (VGG, AlexNet)):
            self.forward_model = self.forward_vgg
            self.backward_model = self.backward_vgg
        elif isinstance(model, (ResNet,)):
            self.forward_model = self.forward_resnet
            self.backward_model = self.backward_resnet
        elif isinstance(model, (GoogLeNet,)):
            self.forward_model = self.forward_googlenet
            self.backward_model = self.backward_googlenet
        elif isinstance(model, (VisionTransformer,)):
            self.forward_model = self.forward_vit
            self.backward_model = self.backward_vit
        else:
            raise Exception(f'{model.__class__} is not available model type')
        self.model = model.cuda()

    def __call__(self, x, yc):
        """
        note that relevance = grad * Delta_x
        """
        # as to increment decomposition, we forward a batch of two inputs
        x0=self.x0
        backward_init=self.backward_init

        with torch.no_grad():
            if x0 is None or x0 == "zero" or x0 == "0" or x0 == 0:
                x0 = torch.zeros_like(x)
            elif x0 == "std0":
                x0 = toStd(torch.zeros_like(x))  # -2.1
            elif x0 == "01n":
                x0 = 0.1 * torch.randn_like(x)
            elif x0 == "03n":
                x0 = 0.3 * torch.randn_like(x)
            elif x0 == "1n":
                x0 = torch.randn_like(x)
            elif x0 == "+n":
                m = x.mean()
                s = x.std()
                assert s >= 0.1, "your input has no discrimination"
                x0 = m + s * torch.randn_like(x)
            else:
                raise Exception()
            x = torch.vstack([x0, x])
            y = self.forward_model(x)

            if isinstance(yc, int):
                yc = torch.tensor([yc], device=y.device)
            elif isinstance(yc, torch.Tensor):
                yc = yc.to(y.device)
            else:
                raise Exception()
            if isinstance(backward_init, torch.Tensor):
                dody = backward_init  # ignore yc
            elif backward_init is None or backward_init == "normal":
                dody = nf.one_hot(yc, y.shape[-1])
            elif backward_init == "negative":
                dody = -nf.one_hot(yc, y.shape[-1])
            elif backward_init == "c":
                dody = nf.one_hot(yc, y.shape[-1]).float()
                dody -= 1 / y.shape[-1]
            elif backward_init == "st":  # st use softmax gradient as initial backward partial derivative
                dody = nf.one_hot(yc, y.shape[-1])
                sm = torch.nn.Softmax(1)
                p = self.forward_baseunit(sm, y)
                dody = self.backward_linearunit(sm, dody)
            elif backward_init == "sig":
                dody = nf.one_hot(yc, y.shape[-1])
                sm = torch.nn.Softmax(1)
                p = self.forward_baseunit(sm, y)
                dody = self.backward_nonlinearunit(sm, dody)
            else:
                raise Exception(f'{backward_init} is not available backward init.')
            g = self.backward_model(dody)
            self.model.x = x
            self.model.g = g
            Rx = x.diff(dim=0) * g
        return g, Rx


if __name__ == '__main__':
    model = get_model(model_names.res50)
    x = get_image_x()
    d = LIDDecomposer(model)
    r = d(x,243)
    show_heatmap(multi_interpolate([r, model.conv1.Ry, model.layer1[-1].relu2.Ry,
                                    model.layer2[-1].relu2.Ry, model.layer3[-1].relu2.Ry,
                                    model.layer4[-1].relu2.Ry]))
    print(r)

"""
# grad check
23.4.28: cuda model gives unusual grad. use cpu model to check grad calculation.
# prepare
self.model = self.model.cpu()
self.x = self.x.cpu()
self.y = self.forward_model(self.x)
self.g = self.backward_model(dody.cpu())
self.x1 = self.x[None,1].detach().requires_grad_()
y = self.model(self.x1)
# easy eval
print((y-self.y[1]).abs().sum())
print((self.x1.grad-self.g).abs().sum())
# clear
self.model = self.model.cuda()
self.x = self.x.cuda()
self.y = self.forward_model(self.x)
self.g = self.backward_model(dody.cuda())


# eval for special model : vgg16
self.clearHooks(self)
i=9 
hookLayerByName(self,self.model,('features',i))
self.model.zero_grad()
with torch.enable_grad():
    self.x1.grad=None
    self.model(self.x1)[0,yc.item()].backward()
print((self.model.features[i].y[1]-self.activation).abs().sum())
print((self.model.features[i].g-self.gradient).abs().sum())


# resnet18
self.clearHooks(self)
hookLayerByName(self,self.model,('layer4',-1,'conv2'))
self.model.zero_grad()
with torch.enable_grad():
    self.x1.grad=None
    self.model(self.x1)[0,yc.item()].backward()
print((self.model.layer4[-1].conv2.y[1]-self.activation).abs().sum())
print((self.model.layer4[-1].conv2.g-self.gradient).abs().sum())

# googlenet
self.clearHooks(self)
hookLayerByName(self,self.model,('conv1','conv'))
self.model.zero_grad()
with torch.enable_grad():
    self.x1.grad=None
    self.model(self.x1)[0,yc.item()].backward()
print((self.model.conv1.conv.y[1]-self.activation).abs().sum())
print((self.model.conv1.conv.g-self.gradient).abs().sum())

# vit
# check encoder
layer=self.model.encoder.layers
x=torch.randn(2,197,768)
with torch.enable_grad():
    x.requires_grad_()
    xs=[x]
    for m in layer:
        y=m(xs[-1])
        y.retain_grad()
        xs.append(y)
    xs[-1].sum().backward()
xxs=[x.clone().detach()]
for m in layer:
    yy=self.forward_encoder_block(m,xxs[-1])
    xxs.append(yy)
gs=[x.grad[1]for x in xs]
ggs=[torch.ones(1,197,768)]
for m in layer[::-1]:
    ggs.append(self.backward_encoder_block(m,ggs[-1]))
ggs=ggs[::-1]
print(torch.tensor([(g-gg).abs().sum() for g,gg in zip(gs,ggs)]))
# finish
self.clearHooks(self)
i=11
hookLayerByName(self,self.model,('encoder','layers',i))
self.model.zero_grad()
with torch.enable_grad():
    self.x.grad=None
    self.model(self.x)[1,yc.item()].backward() # two input
    self.activation=self.activation[None,1]
    self.gradient=self.gradient[None,1]
self.activation=self.activation[0,1:].permute(1,0).reshape(768,14,14)
self.gradient=self.gradient[0,1:].permute(1,0).reshape(768,14,14)
print((self.model.encoder.layers[i].y[1]-self.activation).abs().sum())
print((self.model.encoder.layers[i].g-self.gradient).abs().sum())
"""
