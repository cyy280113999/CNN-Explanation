import torch
import torch.nn.functional as nf
from utils import *

device = 'cuda'


def incr_AvoidNumericInstability(x, eps=1e-9):
    return x + (x >= 0) * eps + (x < 0) * (-eps)


def prop_relev(layer, x, Ry):
    # similar to grad prop
    x = x.clone().detach().requires_grad_()
    y = layer.forward(x)
    Gy = Ry / incr_AvoidNumericInstability(y)
    y.backward(Gy)
    Gx = x.grad
    Rx = x * Gx
    return Rx


def prop_grad(layer, x, Gy):
    x = x.clone().detach().requires_grad_()
    # if you give Ry, then Gy is Ry/y
    # when yi==0, so Ryi==0, it can not know real Gyi
    y = layer.forward(x)
    y.backward(Gy)
    Gx = x.grad
    # Do not return Rx=x*Gx, Gx instead.
    return Gx.clone().detach()


def LRP_Taylor(model, x, yc, device=device, Relevance_Propagate=False):
    layers = ['input layer'] + list(model.features) + [torch.nn.Flatten(1)] + list(model.classifier)
    # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
    # switch relu to non-inplace mode
    for i, layer in enumerate(layers):
        if isinstance(layer, torch.nn.ReLU):
            layers[i].inplace = False
    # flat_loc = len(list(model.features)) + 1
    layerlen = len(layers)

    activations = [None] * layerlen
    x.requires_grad_()
    activations[0] = x
    for i in range(1, layerlen):
        activations[i] = layers[i](activations[i - 1])
    logits = activations[layerlen - 1]

    yc = torch.LongTensor([yc]).detach().to(device)
    target_onehot = nf.one_hot(yc, logits.shape[1]).float().detach()
    if Relevance_Propagate:
        R = [None] * layerlen
        R[layerlen - 1] = target_onehot * activations[layerlen - 1]
        for i in range(1, layerlen)[::-1]:
            R[i - 1] = prop_relev(layers[i], activations[i - 1], R[i])
    else:  # grad prop
        G = [None] * layerlen
        G[layerlen - 1] = target_onehot
        for i in range(1, layerlen)[::-1]:
            G[i - 1] = prop_grad(layers[i], activations[i - 1], G[i])
        R = [a * g for a, g in zip(activations, G)]
    return R
