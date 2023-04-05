import torch
import torch.nn.functional as nf
from torchvision.models import VGG, AlexNet, ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck

from utils import *

"""
LID Decomposer for both linear and nonlinear.

compose linear & nonlinear in one.

use linear=False parameter.

"""


def RelevanceFindByName(model, layer_names=(None,)):
    layer = model
    if not isinstance(layer_names, (tuple, list)):
        layer_names = (layer_names,)
    for l in layer_names:
        if isinstance(l, int):  # for sequential
            layer = layer[l]
        elif isinstance(l, str) and hasattr(layer, l):  # for named child
            layer = layer.__getattr__(l)
        else:
            raise Exception()
    return layer.y.diff(dim=0) * layer.g


def LIDRelevance(model, x, y, layer_names=('features', -1), bp='sig', linear=False):
    d = LIDDecomposer(model, LINEAR=linear, DEFAULT_STEP=11)
    d.forward(x)
    r = d.backward(y, bp)
    return RelevanceFindByName(model, layer_names)


def LID_VGG_m_caller(model, x, y, which_=(23, 30), linear=False, bp='sig'):
    d = LIDDecomposer(model, LINEAR=linear, DEFAULT_STEP=11)
    d.forward(x)
    r = d.backward(y, bp)
    hm = multi_interpolate(RelevanceFindByName('features', i) for i in which_)
    return hm


def LID_Res34_m_caller(model, x, y, which_=(0, 1, 2, 3, 4), linear=False, bp='sig'):
    d = LIDDecomposer(model, LINEAR=linear)
    d.forward(x)
    r = d.backward(y, bp)
    names = ('maxpool', ('layer1', 'relu2'),
             ('layer2', 'relu2'), ('layer3', 'relu2'),
             ('layer4', 'relu2'),)
    hm = multi_interpolate(RelevanceFindByName(names[i]) for i in which_)
    return hm


def LID_Res50_m_caller(model, x, y, which_=(0, 1, 2, 3, 4), linear=False, bp='sig'):
    d = LIDDecomposer(model, LINEAR=linear)
    d.forward(x)
    r = d.backward(y, bp)
    names = ('maxpool', ('layer1', 'relu3'),
             ('layer2', 'relu3'), ('layer3', 'relu3'),
             ('layer4', 'relu3'),)
    hm = multi_interpolate(RelevanceFindByName(names[i]) for i in which_)
    return hm


BaseUnits = (
    torch.nn.Conv2d,
    torch.nn.BatchNorm2d,
    torch.nn.Linear,
    torch.nn.AvgPool2d,
    torch.nn.Flatten,
    torch.nn.ReLU,
    torch.nn.MaxPool2d,
    torch.nn.AdaptiveAvgPool2d,
    torch.nn.Softmax,
)

LinearUnits = (
    torch.nn.Conv2d,
    torch.nn.BatchNorm2d,
    torch.nn.Linear,
    torch.nn.AvgPool2d,
    torch.nn.Flatten,
)

PassUnits = (
    torch.nn.Dropout,  # ignore
)

SpecificUnits = (
    torch.nn.ReLU,  # set inplace to False
    torch.nn.Flatten,  # replaced by manually reshaping
    BasicBlock,  # resnet block
    Bottleneck,  #
)


# SubBlocks = (
#
# )


class LIDDecomposer:
    def forward_baseunit(self, module, x):
        module.x = x
        module.y = module(x)
        self.base_module_saver.append(module)
        return module.y

    def forward_vgg(self, x):
        for i, m in enumerate(self.model.features):
            if hasattr(m, 'inplace'):  # inplace units
                m.inplace = False
            x = self.forward_baseunit(m, x)
        x = self.forward_baseunit(self.model.avgpool, x)
        self.model.last_shape = (1,) + x.shape[1:]
        x = x.flatten(1)
        for m in self.model.classifier:
            if hasattr(m, 'inplace'):
                m.inplace = False
            if isinstance(m, BaseUnits):
                x = self.forward_baseunit(m, x)
            elif isinstance(m, PassUnits):
                pass
            else:
                raise Exception()
        return x

    def backward_linearunit(self, module, g):
        x = module.x[1].unsqueeze(0).clone().detach()
        x.requires_grad_()
        y = module(x)
        (y * g).sum().backward()
        return x.grad.detach()

    def backward_nonlinearunit(self, module, g, step=None):
        if step is None:
            step = self.DEFAULT_STEP
        xs = torch.zeros((step,) + module.x.shape[1:], device=self.DEVICE)
        xs[0] = module.x[0]
        dx = module.x.diff(dim=0) / (step - 1)
        for i in range(1, step):
            xs[i] = xs[i - 1] + dx
        xs.requires_grad_()
        ys = module(xs)
        (ys * g).sum().backward()
        g = xs.grad.mean(0, True).detach()
        return g

    def backward_baseunit(self, m, g):
        m.g = g.detach()
        # m.Ry = m.y.diff(dim=0).detach()
        if self.LINEAR or isinstance(m, LinearUnits):
            g = self.backward_linearunit(m, g)
        else:
            g = self.backward_nonlinearunit(m, g)
        return g

    def backward_vgg(self, g):
        for m in self.model.classifier[::-1]:
            if isinstance(m, BaseUnits):
                g = self.backward_baseunit(m, g)
            elif isinstance(m, PassUnits):
                pass
            else:
                raise Exception()
        g = g.reshape(self.model.last_shape)
        g = self.backward_baseunit(self.model.avgpool, g)
        for m in self.model.features[::-1]:
            g = self.backward_baseunit(m, g)
        return g

    def forward_BasicBlock(self, m, x):
        identity = x
        x = self.forward_baseunit(m.conv1, x)
        x = self.forward_baseunit(m.bn1, x)
        m.relu.inplace = False
        x = self.forward_baseunit(m.relu, x)
        x = self.forward_baseunit(m.conv2, x)
        x = self.forward_baseunit(m.bn2, x)
        if m.downsample is not None:
            for m2 in m.downsample:
                identity = self.forward_baseunit(m2, identity)
        x += identity
        m.relu2 = torch.nn.ReLU(False)
        x = self.forward_baseunit(m.relu2, x)
        return x

    def forward_resnet(self, x):
        x = self.forward_baseunit(self.model.conv1, x)
        x = self.forward_baseunit(self.model.bn1, x)
        self.model.relu.inplace = False
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
        self.model.last_shape = (1,) + x.shape[1:]
        x = x.flatten(1)
        x = self.forward_baseunit(self.model.fc, x)
        return x

    def backward_BasicBlock(self, m, g):
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
        g = g.reshape(self.model.last_shape)
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
        m.relu.inplace = False
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
        return x

    def backward_Bottleneck(self, m, g):
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

    def forward_googlenet(self, x):
        # N x 3 x 224 x 224
        x = self.forward_baseunit(self.model.conv1, x)  # nonlinear
        # N x 64 x 112 x 112
        x = self.forward_baseunit(self.model.maxpool1, x)

        # N x 64 x 56 x 56
        x = self.forward_baseunit(self.model.conv2, x)

        # N x 64 x 56 x 56
        x = self.forward_baseunit(self.model.conv3, x)
        # N x 192 x 56 x 56
        x = self.forward_baseunit(self.model.maxpool2, x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        x = self.forward_baseunit(self.model.conv1, x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        x = self.forward_baseunit(self.model.conv1, x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        x = self.forward_baseunit(self.model.conv1, x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        x = self.forward_baseunit(self.model.conv1, x)
        # N x 512 x 14 x 14

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.forward_baseunit(self.model.avgpool, x)
        # N x 1024 x 1 x 1
        self.model.last_shape = (1,) + x.shape[1:]
        x = x.flatten(1)
        # N x 1024

        x = self.forward_baseunit(self.model.fc, x)
        # N x 1000 (num_classes)
        return x

    def __init__(self, model, LINEAR=False, DEFAULT_STEP=11, DEVICE='cuda'):
        self.DEVICE = DEVICE
        self.LINEAR = LINEAR  # set to nonlinear decomposition
        self.DEFAULT_STEP = DEFAULT_STEP  # step of nonlinear integral approximation
        self.base_module_saver = []
        if isinstance(model, (VGG, AlexNet)):
            # self.model = model.cuda()
            self.forward_model = self.forward_vgg
            self.backward_model = self.backward_vgg
        elif isinstance(model, (ResNet,)):
            # self.model = model.cuda()
            self.forward_model = self.forward_resnet
            self.backward_model = self.backward_resnet
        else:
            raise Exception()
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
                x0 = toStd(torch.zeros_like(x))
            else:
                raise Exception()
            x = torch.vstack([x0, x])
            self.x = x
            self.y = self.forward_model(x)
        return self.y

    def backward(self, yc, backward_init="normal"):
        if isinstance(yc, torch.Tensor):
            yc = yc.item()
        if isinstance(backward_init, torch.Tensor):
            dody = backward_init  # ignore yc
        elif backward_init is None or backward_init == "normal":
            dody = nf.one_hot(torch.tensor([yc], device=self.DEVICE), self.y.shape[-1])
        elif backward_init == "sg":
            dody = nf.one_hot(torch.tensor([yc], device=self.DEVICE), self.y.shape[-1])
            sm = torch.nn.Softmax(1)
            p = self.forward_baseunit(sm, self.y)
            dody = self.backward_linearunit(sm, dody)
        elif backward_init == "sig":
            dody = nf.one_hot(torch.tensor([yc], device=self.DEVICE), self.y.shape[-1])
            sm = torch.nn.Softmax(1)
            p = self.forward_baseunit(sm, self.y)
            dody = self.backward_nonlinearunit(sm, dody)
        else:
            raise Exception()
        self.g = self.backward_model(dody)
        self.Rx = (self.g * (self.x[1] - self.x[0])).detach()
        return self.Rx

    # def __call__(self, x, yc, x0="std0", layer_name=None, backward_init="normal", step=21, device=device):
    #     if layer:
    #         layer = auto_find_layer_name(self.model, layer_name)
    #
    #     if layer is None:
    #         return rys
    #     else:
    #         return rys[layer]


if __name__ == '__main__':
    interpolate_to_imgsize = lambda x: heatmapNormalizeR(nf.interpolate(x.sum(1, True), 224, mode='bilinear'))
    multi_interpolate = lambda xs: heatmapNormalizeR(
        sum(heatmapNormalizeR(nf.interpolate(x.sum(1, True), 224, mode='bilinear')) for x in xs))

    model = get_model('resnet50')
    x = get_image_x()
    d = LIDDecomposer(model)
    d.forward(x)
    r = d.backward(243)
    showHeatmap(multi_interpolate([r, model.conv1.Ry, model.layer1[-1].relu2.Ry,
                                   model.layer2[-1].relu2.Ry, model.layer3[-1].relu2.Ry,
                                   model.layer4[-1].relu2.Ry]))
    print(r)
