"""
layer-wise linear decomposition:

LRP_Taylor
"""

import torch.nn.functional as nf
from utils import *
from methods.LRP import softmax_gradient


def incr_AvoidNumericInstability(x, eps=1e-9):
    return x + (x >= 0) * eps + (x < 0) * (-eps)


def prop_relev(x, x0, layer, Ry):
    x = x.clone().detach().requires_grad_()
    dx = x - x0
    y = layer.forward(x)
    y0 = layer.forward(x0)
    dy = y - y0
    dody = Ry / incr_AvoidNumericInstability(dy)
    y.backward(dody)
    dodx = x.grad
    Rx = dx * dodx

    # Gy = Ry / incr_AvoidNumericInstability(y)
    # y.backward(Gy)
    # Gx=x.grad
    # Rx=x * Gx
    return Rx.clone().detach()


class LIDLinearDecomposer:
    def __init__(self, model):
        # layers is the used vgg
        self.model = model
        assert isinstance(model, (torchvision.models.VGG,
                                  torchvision.models.AlexNet))
        self.layers = ['x layer'] + list(model.features) + [torch.nn.Flatten(1)] + list(model.classifier)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.ReLU):
                self.layers[i] = torch.nn.ReLU(inplace=False)
        self.flat_loc = 1 + len(list(model.features))
        self.layerlen = len(self.layers)

    def __call__(self, x, yc, x0="std0", layer=None, backward_init="normal", device=device, Relevance_Propagate = False):
        if layer:
            layer = auto_find_layer_index(self.model, layer)
        if x0 is None or x0 == "zero":
            x0 = torch.zeros_like(x)
        elif x0 == "std0":
            x0 = toStd(torch.zeros_like(x))
        else:
            raise Exception()

        # forward
        ys = [None] * self.layerlen  # store each layer output ,layer0 is specific input layer
        y0s = [None] * self.layerlen
        ys[0] = x.requires_grad_()  # layer0 output is x
        y0s[0] = x0
        for i in range(1, self.layerlen):
            ys[i] = self.layers[i](ys[i - 1])
            ys[i].retain_grad()
            y0s[i] = self.layers[i](y0s[i - 1])

        dys = [y - y0 for y, y0 in zip(ys, y0s)]

        # backward
        if isinstance(yc, torch.Tensor):
            yc = yc.item()
        # in LinearDecomposition , we represent dody as backward init instead of relevance,
        # so ST-LRP_0 is equal to SG_LRP_Taylor
        if isinstance(backward_init, torch.Tensor):
            dody = backward_init  # ignore yc
        elif backward_init is None or backward_init == "normal":
            dody = nf.one_hot(torch.tensor([yc], device=device), dys[-1].shape[1])
        elif backward_init == "sg":
            dody = softmax_gradient(nf.softmax(ys[-1], dim=1), target_class=yc)
        elif backward_init == "sig":
            avggrad = torch.zeros_like(ys[-1])
            lin_samples = torch.linspace(0, 1, 11, device=device)
            for scale_multiplier in lin_samples:
                avggrad += softmax_gradient(nf.softmax(scale_multiplier * ys[-1], dim=1), target_class=yc)
            avggrad /= 11
            dody = avggrad
        else:
            raise Exception()

        if Relevance_Propagate:
            rys = [None] * self.layerlen
            rys[-1] = dody * dys[-1]
            for i in range(1, self.layerlen)[::-1]:
                rys[i - 1] = prop_relev(ys[i - 1], y0s[i - 1], self.layers[i], rys[i])
        else:  # grad prop
            (dody * dys[-1]).sum().backward()
            rys = [y.grad * dy for y, dy in zip(ys, dys)]
        if layer is None:
            return rys
        else:
            return rys[layer]


if __name__ == '__main__':
    model = get_model()
    filename = '../testImg.png'
    x = pilOpen(filename)
    x = pilToTensor(x).unsqueeze(0)
    x = toStd(x).to(device)
    hm = LIDLinearDecomposer(model)(x, 243)
    print(hm)
