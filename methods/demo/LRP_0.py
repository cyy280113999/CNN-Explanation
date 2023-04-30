"""
LRP-0 demo for beginning user.


"LRP-0-f": lambda model: lambda x, y: interpolate_to_imgsize(
    LRP_0(model, x, y)[31]),

"""


import torch
import torch.nn.functional as nf
from torchvision.models import VGG

from utils import *



device = 'cuda'


def safeDivide(x, y, eps=1e-9):
    return (x / (y + y.ge(0) * eps + y.lt(0) * (-eps))) * (y.abs() > eps)


def LRP_0_layer(layer, x, Ry):
    # follow the easy implement of lrp-overview
    if isinstance(layer, torch.nn.Linear):
        w = layer.weight
        y = layer.forward(x)  # y=xwT+b
        s = safeDivide(Ry, y)  # means Gy
        c = s.mm(w)  # means Gx
    elif isinstance(layer, torch.nn.Conv2d):
        w = layer.weight
        y = layer.forward(x)  # conv(x)
        s = safeDivide(Ry, y)
        c = torch.conv_transpose2d(
            s, w, bias=None, stride=layer.stride, padding=layer.padding,
            groups=layer.groups, dilation=layer.dilation)
    else:
        raise Exception()
    Rx = c * x
    return Rx


def LRP_Taylor_layer(layer, x, Ry):
    x = x.clone().detach().requires_grad_()
    y = layer.forward(x)
    Gy = safeDivide(Ry, y)
    y.backward(Gy)
    Gx = x.grad
    Rx = x * Gx
    return Rx


LRP_0_Units = (
    torch.nn.Linear,
    torch.nn.Conv2d,
)

LRP_Taylor_Units = (
    torch.nn.MaxPool2d,
)

LRP_Specific_Units = (
    torch.nn.Flatten,
)


def LRP_0(model, x, yc, device=device, Relevance_Propagate=False):
    # -- vgg16 only!! vgg is a classic sequential model
    assert isinstance(model, VGG)
    # create model, add input layer. so the count of layer added by one
    layers = ['input layer'] + list(model.features) + [torch.nn.Flatten(1)] + list(model.classifier)
    layerlen = len(layers)

    # forward
    activations = [None] * layerlen
    x.requires_grad_()
    activations[0] = x
    for i in range(1, layerlen):
        activations[i] = layers[i](activations[i - 1])
    logits = activations[layerlen - 1]

    # backward
    yc = torch.LongTensor([yc]).detach().to(device)
    target_onehot = nf.one_hot(yc, logits.shape[1]).float().detach()
    R = [None] * layerlen
    R[layerlen - 1] = target_onehot * activations[layerlen - 1]
    for i in range(1, layerlen)[::-1]:
        if isinstance((layers[i]),torch.nn.Flatten):
            R[i-1] = R[i].reshape_as(activations[i-1])
        elif isinstance(layers[i], LRP_Taylor_Units):
            R[i - 1] = LRP_Taylor_layer(layers[i], activations[i - 1], R[i])
        elif isinstance(layers[i], LRP_0_Units):
            R[i - 1] = LRP_0_layer(layers[i], activations[i - 1], R[i])
        else:
            R[i - 1] = R[i]
    return R


if __name__ == '__main__':
    model = get_model()
    filename = 'testImg.png'
    x = pilOpen(filename)
    x = toTensorS224(x).unsqueeze(0)
    x = toStd(x).to(device)
    hm = LRP_0(model, x, 243)
    show_heatmap(hm)
