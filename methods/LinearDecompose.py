
"""
linear decomposition:

LRP_Taylor
"""

import torch.nn.functional as nf
from utils import *

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


class LinearDecomposer:
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

    def __call__(self, x, dody_or_yc=None, x0=None,layer=None, device=device):
        layer = auto_find_layer_index(self.model, layer)
        # forward
        ys = [None] * self.layerlen  # store each layer output ,layer0 is specific input layer
        y0s = [None] * self.layerlen
        ys[0] = x.requires_grad_()  # layer0 output is x
        y0s[0] = torch.zeros_like(x) if x0 is None else x0
        for i in range(1, self.layerlen):
            ys[i] = self.layers[i](ys[i - 1])
            ys[i].retain_grad()
            y0s[i] = self.layers[i](y0s[i - 1])

        dys = [y - y0 for y, y0 in zip(ys, y0s)]

        # backward
        if dody_or_yc is None:
            dody = nf.one_hot(torch.tensor([0], device=device), dys[-1].shape[1])
        elif isinstance(dody_or_yc, int):
            dody = nf.one_hot(torch.tensor([dody_or_yc], device=device), dys[-1].shape[1])
        elif isinstance(dody_or_yc, torch.Tensor):
            dody = dody_or_yc.detach()
        else:
            raise Exception()

        Relevance_Propagate = False
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
    x=pilOpen(filename)
    x = pilToTensor(x).unsqueeze(0)
    x = toStd(x).to(device)
    hm=LinearDecomposer(model)(x,243)
    print(hm)
