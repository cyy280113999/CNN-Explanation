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


# this give special layer heatmap
def LID_caller(model, x, y, x0='std0', layer_name=('features', -1), bp='ag', linear=False):
    d = LIDDecomposer(model, LINEAR=linear, DEFAULT_STEP=11)
    d(x, y, x0, bp)
    hm = interpolate_to_imgsize(relevanceFindByName(model, layer_name))
    return hm


# this gives the stage wrapper for common nets
# and multi-layer mixed heatmap
def LID_m_caller(model, x, y, x0='std0', which_=(0, 1, 2, 3, 4, 5), linear=False, bp='ag'):
    if not isinstance(which_, (list, tuple)):
        which_ = (which_,)
    if isinstance(model, VGG):  # None for Rx
        layer_names = [('features', i) for i in (0, 4, 9, 16, 23, 30)]
    elif isinstance(model, AlexNet):
        layer_names = [('features', i) for i in (0, 2, 5, 12)]
    elif isinstance(model, ResNet):
        layer_names = ['conv1', 'maxpool'] + [(f'layer{i}', -1) for i in (1, 2, 3, 4)]
    elif isinstance(model, GoogLeNet):  # None for Rx
        layer_names = ['conv1', 'maxpool1', 'maxpool2',
                       'inception3b', 'inception4e', 'inception5b']
    else:
        raise Exception(f'{model.__class__} is not available model type')
    d = LIDDecomposer(model, LINEAR=linear, DEFAULT_STEP=11)
    d(x, y, x0, bp)
    hm = multi_interpolate(relevanceFindByName(model, layer_names[i]) for i in which_)
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
ST-LRP=SG-LID
    "ST-LRP-0-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='st', method='lrp0', layer_num=-1)),
    "SG-LID-LRP-0-f": lambda model: lambda x, y: LRP_caller(model, x, y, layer_name=('features', -1), bp='sg'),
    "ST-LRP-0-24": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='st', method='lrp0', layer_num=24)),
    "SG-LID-LRP-0-23": lambda model: lambda x, y: LRP_caller(model, x, y, layer_name=('features', 23), bp='sg'),
"""
def LRP_caller(model, x, y, x0='std0', layer_name=('features', -1), bp='ag'):
    d = LIDDecomposer(model, LINEAR=True)
    d(x, y, x0, bp)
    if layer_name == 'input_layer' or layer_name[0] == 'input_layer':
        r = model.x[1] * model.gx
    else:
        layer = findLayerByName(model, layer_name)
        r = layer.y[1] * layer.g
    hm = interpolate_to_imgsize(r)
    return hm


"""
like RelevanceCAM, reducing grid pattern
"""
#     "LID-CAM-sg": lambda model: lambda x, y: interpolate_to_imgsize(
#         LID_CAM(model, x, y, x0='std0', layer_name=('layer1', -1, 'relu2'), bp='sg', linear=False)),
#     "LID-CAM-c": lambda model: lambda x, y: interpolate_to_imgsize(
#         LID_CAM(model, x, y, x0='std0', layer_name=('layer1', -1, 'relu2'), bp='c', linear=False)),
#     "LID-CAM-1": lambda model: lambda x, y: interpolate_to_imgsize(
#         LID_CAM(model, x, y, x0='std0', layer_name=('layer1', -1, 'relu2'), bp='ag', linear=False)),
# def LID_CAM(model, x, y, x0='std0', layer_name=('features', -1), bp='ag', linear=False):
#     d = LIDDecomposer(model, LINEAR=linear, DEFAULT_STEP=11)
#     d(x, y, x0, bp)
#     layer = findLayerByName(model, layer_name)
#     r = layer.y.diff(dim=0) * layer.g
#     return (r.sum([2, 3], True) * layer.y[1]).sum(1, True)


"""
this give pixel level image
"""
# def LID_image(model, x, y, x0='std0', bp='ag', linear=False):
#     d = LIDDecomposer(model, LINEAR=linear, DEFAULT_STEP=11)
#     g, rx = d(x, y, x0, bp)
#     rx = heatmapNormalizeR(rx.detach().cpu().clip(min=0))  # how to use negative?
#     # r = d.backward(y, bp).detach().cpu()
#     # stdr=r.std(dim=[2,3], keepdim=True)
#     # r = ImgntStdTensor/stdr * r
#     # r = invStd(r)
#     return rx


# def LID_grad(model, x, y, x0='std0', bp='ag', linear=False):
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
    # nonlinear
    torch.nn.MaxPool2d,
    torch.nn.AdaptiveAvgPool2d,
    torch.nn.Softmax,
)

LinearUnits = (
    torch.nn.Conv2d,
    torch.nn.BatchNorm2d,
    torch.nn.Linear,
    torch.nn.AvgPool2d,
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


class LIDDecomposer:
    def forward_baseunit(self, m, x):
        if not isinstance(m, BaseUnits):
            raise Exception(f'layer:{m} is not a supported base layer')
        if isinstance(m, torch.nn.ReLU):
            m.inplace = False
        elif isinstance(m, torch.nn.Dropout):
            return x
        m.x = x
        m.y = m(x)
        return m.y

    def forward_vgg(self, x):
        for i, m in enumerate(self.model.features):
            x = self.forward_baseunit(m, x)
        x = self.forward_baseunit(self.model.avgpool, x)
        self.model.flatten = torch.nn.Flatten()
        x = self.forward_baseunit(self.model.flatten, x)
        for m in self.model.classifier:
            x = self.forward_baseunit(m, x)
        return x

    def backward_linearunit(self, module, g):
        with torch.enable_grad():
            x = module.x[1].unsqueeze(0).detach().requires_grad_()
            y = module(x)
            (y * g).sum().backward()
        return x.grad

    def backward_nonlinearunit(self, module, g, step=None):
        if step is None:
            step = self.DEFAULT_STEP
        xs = torch.zeros((step,) + module.x.shape[1:], device=self.DEVICE)
        xs[0] = module.x[0]
        dx = module.x.diff(dim=0) / (step - 1)
        for i in range(1, step):
            xs[i] = xs[i - 1] + dx
        with torch.enable_grad():
            xs.requires_grad_()
            ys = module(xs)
            (ys * g).sum().backward()
        g = xs.grad.mean(0, True).detach()
        return g

    def backward_baseunit(self, m, g):
        m.g = g.detach()
        # m.Ry = m.y.diff(dim=0).detach()
        if isinstance(m, torch.nn.Flatten):
            return g.reshape((1,) + m.x.shape[1:])
        elif isinstance(m, torch.nn.Dropout):
            return g
        elif self.LINEAR or isinstance(m, LinearUnits):
            g = self.backward_linearunit(m, g)
        elif isinstance(m, BaseUnits):  # nonlinear
            g = self.backward_nonlinearunit(m, g)
        else:
            raise Exception()
        return g

    def backward_vgg(self, g):
        for m in self.model.classifier[::-1]:
            g = self.backward_baseunit(m, g)
        g = self.backward_baseunit(self.model.flatten, g)
        g = self.backward_baseunit(self.model.avgpool, g)
        for m in self.model.features[::-1]:
            g = self.backward_baseunit(m, g)
        return g

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

    def backward_BasicBlock(self, m, g):
        m.g = g  # save for block relevance
        g = self.backward_baseunit(m.relu2, g)
        out_g = g
        if m.downsample is not None:
            for m2 in m.downsample[::-1]:
                out_g = self.backward_baseunit(m2, out_g)
        g = self.backward_baseunit(m.bn2, g)
        g = self.backward_baseunit(m.conv2, g)
        g = self.backward_baseunit(m.relu, g)
        g = self.backward_baseunit(m.bn1, g)
        g = self.backward_baseunit(m.conv1, g)
        g += out_g
        return g

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

    def forward_Bottleneck(self, m, x):
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

    def backward_Bottleneck(self, m, g):
        m.g = g
        g = self.backward_baseunit(m.relu3, g)
        out_g = g
        if m.downsample is not None:
            for sub_m in m.downsample[::-1]:
                out_g = self.backward_baseunit(sub_m, out_g)
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

    def forward_BasicConv2d(self, m, x):
        x = self.forward_baseunit(m.conv, x)
        x = self.forward_baseunit(m.bn, x)
        m.relu = torch.nn.ReLU(False)
        x = self.forward_baseunit(m.relu, x)
        return x

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

    def backward_BasicConv2d(self, m, g):
        g = self.backward_baseunit(m.relu, g)
        g = self.backward_baseunit(m.bn, g)
        g = self.backward_baseunit(m.conv, g)
        return g

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

    def forward_self_attention(self, m, x):
        []
        return x

    def forward_encoder_block(self, m, x):
        tmp = x
        x = self.forward_baseunit(m.ln_1, x)
        x, _ = m.self_attention(x, x, x, need_weights=False)
        x = x + tmp
        tmp = x
        x = self.forward_baseunit(m.ln_2, x)
        x = m.mlp(x)
        x = x + tmp
        return x

    def forward_vit(self, x):
        assert isinstance(self.model, VisionTransformer)
        n, c, h, w = x.shape
        assert h == 224
        assert self.model.patch_size == 16
        # p = self.model.patch_size  # 16
        # n_h = h // p  # 14
        # n_w = w // p
        x = self.forward_baseunit(self.model.conv_proj, x)
        x = x.reshape(n, self.model.hidden_dim, 196)
        x = x.permute(0, 2, 1)
        batch_class_token = self.model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
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
        tmp = torch.zeros_like(self.model.encoder.ln.y[1, None].shape)
        tmp[:, 0] = g
        g = tmp
        self.model.encoder.ln
        for m in self.model.encoder.layers[::-1]:
            []
        raise NotImplementedError()

    def __init__(self, model, LINEAR=False, DEFAULT_STEP=11, DEVICE='cuda'):
        self.DEVICE = DEVICE
        self.LINEAR = LINEAR  # set to nonlinear decomposition
        self.DEFAULT_STEP = DEFAULT_STEP  # step of nonlinear integral approximation
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

    # def clean(self):
    #     for m in self.base_module_saver:
    #         m.x=None
    #         m.y=None
    #         m.Ry=None

    def forward(self, x, x0="std0"):
        # as to increment decomposition, we forward a batch of two inputs
        with torch.no_grad():
            if x0 is None or x0 == "zero":
                x0 = torch.zeros_like(x)
            elif x0 == "std0":
                x0 = toStd(torch.zeros_like(x))  # -2.1
            elif x0 == "01n":
                x0 = 0.1 * torch.randn_like(x)
            elif x0 == "03n":
                x0 = 0.3 * torch.randn_like(x)
            else:
                raise Exception()
            self.x = torch.vstack([x0, x])
            self.model.x = self.x
            self.y = self.forward_model(self.x)
        return self.y

    def backward(self, yc, backward_init="normal"):
        """
        note that relevance = grad * Delta_x
        """
        with torch.no_grad():
            if isinstance(yc, int):
                yc = torch.tensor([yc], device=self.y.device)
            elif isinstance(yc, torch.Tensor):
                yc = yc.to(self.y.device)
            else:
                raise Exception()
            if isinstance(backward_init, torch.Tensor):
                dody = backward_init  # ignore yc
            elif backward_init is None or backward_init == "normal":
                dody = nf.one_hot(yc, self.y.shape[-1])
            elif backward_init == "negative":
                dody = -nf.one_hot(yc, self.y.shape[-1])
            elif backward_init == "c":
                dody = nf.one_hot(yc, self.y.shape[-1]).float()
                dody -= 1 / self.y.shape[-1]
            elif backward_init == "sg":  # sg use softmax gradient as initial backward partial derivative
                dody = nf.one_hot(yc, self.y.shape[-1])
                sm = torch.nn.Softmax(1)
                p = self.forward_baseunit(sm, self.y)
                dody = self.backward_linearunit(sm, dody)
            elif backward_init == "ag":
                dody = nf.one_hot(yc, self.y.shape[-1])
                sm = torch.nn.Softmax(1)
                p = self.forward_baseunit(sm, self.y)
                dody = self.backward_nonlinearunit(sm, dody)
            else:
                raise Exception(f'{backward_init} is not available backward init.')
            self.g = self.backward_model(dody)
            self.model.gx = self.g
            self.Rx = self.x.diff(dim=0) * self.g
        return self.g, self.Rx

    def __call__(self, x, yc, x0="std0", backward_init="normal"):
        self.forward(x, x0)
        self.backward(yc, backward_init)
        return self.g, self.Rx


if __name__ == '__main__':
    model = get_model('resnet50')
    x = get_image_x()
    d = LIDDecomposer(model)
    d.forward(x)
    g, r = d.backward(243)
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
i=9 
self.clearHooks(self)
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


"""
