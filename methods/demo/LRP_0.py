import torch
import torch.nn.functional as nf
from utils import *

# "LRP-0-f-grad": lambda model: lambda x, y: interpolate_to_imgsize(
#     LRP_0(model, x, y, Relevance_Propagate=False)[31]),
# "LRP-0-f-relev": lambda model: lambda x, y: interpolate_to_imgsize(
#     LRP_0(model, x, y, Relevance_Propagate=True)[31]),

device = 'cuda'


def incr_AvoidNumericInstability(x, eps=1e-9):
    return x + (x >= 0) * eps + (x < 0) * (-eps)


def prop_relev_demo(layer, x, Ry):
    # -- Original Relevance Propagation
    # This demo is to demonstrate how the relevance propagate.
    # If you use this jacobian version, will get "CUDA out of memory" in common CNNs.
    # we move dim of x to the end, y to the start. dim of x&y is seperated.
    x = x.squeeze(0)  # remove batch
    Ry = Ry.squeeze(0)
    x_dim_depth = len(x.shape)
    x_empty_dim = (1,) * x_dim_depth
    y = layer.forward(x)
    y_dim_depth = len(y.shape)
    y = y.reshape(y.shape + x_empty_dim)  # y as (y_shape, 1,..,1)
    # we get the jacobian whose dim match x&y
    # on FC layer , you will see jacobian == layer.weight
    g = torch.autograd.functional.jacobian(lambda x: layer.forward(x), x)
    # we use jacobian to approximate the increment of output
    # weight=g*y/incr_AvoidNumericInstability(x)
    # r=Ry*weight
    r = Ry * g * y / incr_AvoidNumericInstability(x)
    Rx = r.sum(list(range(y_dim_depth)))  # sum according y_shape
    return Rx.unsqueeze(0)


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


LRP_0_Units = (
    torch.nn.Linear,
    torch.nn.Conv2d,
    torch.nn.MaxPool2d,
)


def LRP_0(model, x, yc, device=device, Relevance_Propagate=False):
    # -- vgg16 only!!
    # create model
    layers = ['input layer'] + list(model.features) + [torch.nn.Flatten(1)] + list(model.classifier)

    # flat_loc = len(list(model.features)) + 1
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
    if Relevance_Propagate:
        R = [None] * layerlen
        R[layerlen - 1] = target_onehot * activations[layerlen - 1]
        for i in range(1, layerlen)[::-1]:
            if isinstance(layers[i], LRP_0_Units):
                R[i - 1] = prop_relev(layers[i], activations[i - 1], R[i])
            else:
                R[i - 1] = R[i]
    else:  # grad prop
        G = [None] * layerlen
        G[layerlen - 1] = target_onehot
        for i in range(1, layerlen)[::-1]:
            if isinstance(layers[i], LRP_0_Units):
                G[i - 1] = prop_grad(layers[i], activations[i - 1], G[i])
            else:
                G[i - 1] = G[i]
        R = [a * g for a, g in zip(activations, G)]
    return R





if __name__ == '__main__':
    model = get_model()
    filename = '../testImg.png'
    x = pilOpen(filename)
    x = toTensorS224(x).unsqueeze(0)
    x = toStd(x).to(device)
    hm = LRP_0(model, x, 243)
    print(hm)
