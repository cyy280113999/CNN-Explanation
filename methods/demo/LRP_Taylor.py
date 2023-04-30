"""
LRP-Taylor equals to LRP-0

"LRP-Taylor-f": lambda model: lambda x, y: interpolate_to_imgsize(
    LRP_Taylor(model, x, y)[31]),
"""
import torch
import torch.nn.functional as nf
from utils import *

device = 'cuda'


def safeDivide(x, y, eps=1e-9):
    return (x / (y + y.ge(0) * eps + y.lt(0) * (-eps))) * (y.abs() > eps)

# lrp taylor works for all kinds of layer
def LRP_Taylor_layer(layer, x, Ry):
    x = x.clone().detach().requires_grad_()
    y = layer.forward(x)
    Gy = safeDivide(Ry, y)
    y.backward(Gy)
    Gx = x.grad
    Rx = x * Gx
    return Rx


def prop_grad(layer, x, Gy):
    x = x.clone().detach().requires_grad_()
    y = layer.forward(x)
    y.backward(Gy)
    Gx = x.grad
    return Gx.clone().detach()


def LRP_Taylor(model, x, yc, device=device):
    layers = ['input layer'] + list(model.features) + [torch.nn.Flatten(1)] + list(model.classifier)
    # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
    # switch relu to non-inplace mode
    for i, layer in enumerate(layers):
        if isinstance(layer, torch.nn.ReLU):
            layers[i].inplace = False
    layerlen = len(layers)

    activations = [None] * layerlen
    x.requires_grad_()
    activations[0] = x
    for i in range(1, layerlen):
        activations[i] = layers[i](activations[i - 1])
    logits = activations[layerlen - 1]

    yc = torch.LongTensor([yc]).detach().to(device)
    target_onehot = nf.one_hot(yc, logits.shape[1]).float().detach()
    R = [None] * layerlen
    R[layerlen - 1] = target_onehot * activations[layerlen - 1]
    for i in range(1, layerlen)[::-1]:
        R[i - 1] = LRP_Taylor_layer(layers[i], activations[i - 1], R[i])
    return R

if __name__ == '__main__':
    model = get_model()
    filename = 'testImg.png'
    x = pilOpen(filename)
    x = toTensorS224(x).unsqueeze(0)
    x = toStd(x).to(device)
    hm = LRP_Taylor(model, x, 243)
    show_heatmap(hm)
