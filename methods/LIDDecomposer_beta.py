import torch
import torch.nn.functional as nf
from torchvision.models import VGG, AlexNet, ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck

from utils import *
"""
this can not output relevance of middle layer , but it saved in middle_layer.Ry

support resnet
"""

device = 'cuda'

BaseUnits = (
    torch.nn.Conv2d,
    torch.nn.BatchNorm2d,
    torch.nn.Linear,
    torch.nn.AvgPool2d,
    torch.nn.Flatten,
    torch.nn.ReLU,
    torch.nn.MaxPool2d,
    torch.nn.AdaptiveAvgPool2d,
    # torch.nn.Softmax,
)

LinearUnits=(
    torch.nn.Conv2d,
    torch.nn.BatchNorm2d,
    torch.nn.Linear,
    torch.nn.AvgPool2d,
    torch.nn.Flatten,
)

PassUnits=(
    torch.nn.Dropout,  # ignore
)

SpecificUnits=(
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
            if hasattr(m,'inplace'):  # inplace units
                m.inplace=False
            x = self.forward_baseunit(m, x)
        x = self.forward_baseunit(self.model.avgpool, x)
        self.model.last_shape = (1,)+x.shape[1:]
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
        x = module.x[0].unsqueeze(0).clone().detach()
        x.requires_grad_()
        y = module(x)
        (y * g).sum().backward()
        return x.grad.detach()

    def backward_nonlinearunit(self, module, g, step=None):
        if step is None:
            step = self.DEFAULT_STEP
        xs = torch.zeros((step,) + module.x.shape[1:], device=self.DEVICE)
        xs[0] = module.x[0]
        dx = (module.x[1] - module.x[0]) / (step - 1)
        for i in range(1, step):
            xs[i] = xs[i - 1] + dx
        xs.requires_grad_()
        ys = module(xs)
        (ys * g).sum().backward()
        g = xs.grad.mean(0, True).detach()
        return g

    def backward_baseunit(self, m, g):
        m.Ry = ((m.y[1] - m.y[0]) * g).detach().cpu()
        if isinstance(m,LinearUnits):
            g=self.backward_linearunit(m,g)
        else:
            g=self.backward_nonlinearunit(m,g)
        return g

    def backward_vgg(self, g):
        for m in self.model.classifier[::-1]:
            if isinstance(m,BaseUnits):
                g = self.backward_baseunit(m, g)
            elif isinstance(m,PassUnits):
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
        m.relu.inplace=False
        x = self.forward_baseunit(m.relu,x)
        x = self.forward_baseunit(m.conv2, x)
        x = self.forward_baseunit(m.bn2, x)
        if m.downsample is not None:
            for m2 in m.downsample:
                identity = self.forward_baseunit(m2, identity)
        x += identity
        m.relu2=torch.nn.ReLU(False)
        x = self.forward_baseunit(m.relu2,x)
        return x

    def forward_resnet(self, x):
        x = self.forward_baseunit(self.model.conv1, x)
        x = self.forward_baseunit(self.model.bn1, x)
        self.model.relu.inplace=False
        x = self.forward_baseunit(self.model.relu, x)
        x = self.forward_baseunit(self.model.maxpool, x)
        if isinstance(self.model.layer1[0],BasicBlock):
            self.forward_block=self.forward_BasicBlock
            self.backward_block = self.backward_BasicBlock
        else:
            self.forward_block=self.forward_Bottleneck
            self.backward_block=self.backward_Bottleneck
        for m in self.model.layer1:
            x = self.forward_block(m, x)
        for m in self.model.layer2:
            x = self.forward_block(m, x)
        for m in self.model.layer3:
            x = self.forward_block(m, x)
        for m in self.model.layer4:
            x = self.forward_block(m, x)
        x = self.forward_baseunit(self.model.avgpool, x)
        self.model.last_shape=(1,)+x.shape[1:]
        x = x.flatten(1)
        x = self.forward_baseunit(self.model.fc, x)
        return x

    def backward_BasicBlock(self, m, g):
        g = self.backward_baseunit(m.relu2,g)
        out_g=g
        if m.downsample is not None:
            for m2 in m.downsample[::-1]:
                out_g = self.backward_baseunit(m2,out_g)
        g = self.backward_baseunit(m.bn2, g)
        g = self.backward_baseunit(m.conv2, g)
        g = self.backward_baseunit(m.relu,g)
        g = self.backward_baseunit(m.bn1, g)
        g = self.backward_baseunit(m.conv1, g)
        g+=out_g
        return g

    def backward_resnet(self, g):
        g = self.backward_baseunit(self.model.fc, g)
        g=g.reshape(self.model.last_shape)
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

    def forward_Bottleneck(self,m,x):
        identity = x
        x=self.forward_baseunit(m.conv1,x)
        x = self.forward_baseunit(m.bn1, x)
        m.relu.inplace=False
        x = self.forward_baseunit(m.relu, x)
        x = self.forward_baseunit(m.conv2, x)
        x = self.forward_baseunit(m.bn2, x)
        m.relu2=torch.nn.ReLU(False)
        x = self.forward_baseunit(m.relu2, x)
        x = self.forward_baseunit(m.conv3, x)
        x = self.forward_baseunit(m.bn3, x)
        if m.downsample is not None:
            for sub_m in m.downsample:
                identity = self.forward_baseunit(sub_m,identity)
        x += identity
        m.relu3=torch.nn.ReLU(False)
        x = self.forward_baseunit(m.relu3,x)
        return x

    def backward_Bottleneck(self,m,g):
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


    DEVICE = torch.device('cuda')

    def __init__(self, model, DEFAULT_STEP=5):
        self.DEFAULT_STEP = DEFAULT_STEP
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

    def __del__(self):
        self.clean()

    def clean(self):
        for m in self.base_module_saver:
            m.x=None
            m.y=None
            m.Ry=None

    def forward(self, x, x0="std0"):
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
            dody = nf.one_hot(torch.tensor([yc], device=device), self.y.shape[-1])
        else:
            raise Exception()
        self.g = self.backward_model(dody)
        self.Rx = self.g * (self.x[1]-self.x[0])
        return self.Rx

    # def __call__(self, x, yc, x0="std0", layer=None, backward_init="normal", step=21, device=device):
    #     if layer:
    #         layer = auto_find_layer_index(self.model, layer)
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
    filename = '../testImg.png'
    x = pilOpen(filename)
    x = toTensorS224(x).unsqueeze(0)
    x = toStd(x).to(device)
    d=LIDDecomposer(model)
    d.forward(x)
    r=d.backward(243)
    showHeatmap(multi_interpolate([r, model.conv1.Ry, model.layer1[-1].relu2.Ry,
                                   model.layer2[-1].relu2.Ry, model.layer3[-1].relu2.Ry,
         model.layer4[-1].relu2.Ry]))
    print(r)