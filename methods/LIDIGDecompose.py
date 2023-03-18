
"""
layer-wise integrated gradient decomposition

LID-IG Decompose
"""

import torch.nn.functional as nf
from utils import *


def IG_prop_grad(x,x0,layer,gy,step=11):
    xs=[]
    dx = (x - x0) / (step-1)  # step include start and end
    for i in range(0, step):
        xs.append(x0 + i * dx)
    xs = torch.vstack(xs).detach().requires_grad_()
    ys = layer(xs)
    (ys*gy).sum().backward()
    return xs.grad.mean(0,True)

def IG_prop_relev(x,x0,layer,ry,step=11):
    xs=[]
    dx = (x - x0) / step
    for i in range(0, step):
        xs.append(x0 + i * dx)
    xs = torch.vstack(xs).requires_grad_()
    ys = layer(xs)

    raise Exception()


class LIDIGDecomposer:
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

    def __call__(self, x, yc, x0="std0", layer=None, backward_init="normal", step=11, device=device):
        if layer:
            layer = auto_find_layer_index(self.model, layer)
        # forward
        ys = [None] * self.layerlen
        y0s = [None] * self.layerlen
        ys[0] = x.requires_grad_()
        if x0 is None or x0=="zero":
            y0s[0] = torch.zeros_like(x)
        elif x0=="std0":
            y0s[0] = toStd(torch.zeros_like(x))
        else:
            raise Exception()
        for i in range(1, self.layerlen):
            ys[i] = self.layers[i](ys[i - 1])
            y0s[i] = self.layers[i](y0s[i - 1])

        dys = [y-y0 for y,y0 in zip(ys, y0s)]

        # backward
        if isinstance(yc,torch.Tensor):
            yc=yc.item()
        if isinstance(backward_init,torch.Tensor):
            dody=backward_init  # ignore yc
        elif backward_init is None or backward_init == "normal":
            dody = nf.one_hot(torch.tensor([yc], device=device), dys[-1].shape[1])
        elif backward_init == "sig":
            dody = nf.one_hot(torch.tensor([yc], device=device), dys[-1].shape[1])
            dody = IG_prop_grad(ys[-1],y0s[-1],lambda x:nf.softmax(x,1),dody,step=step)
        else:
            raise Exception()
        _stop_at = layer if layer is not None else 0
        gys = [None] * self.layerlen
        gys[-1] = dody
        for i in range(_stop_at+1, self.layerlen)[::-1]:
            gys[i - 1] = IG_prop_grad(ys[i-1],y0s[i-1],self.layers[i],gys[i],step=step)
        rys = [gy * dy for gy, dy in zip(gys, dys)]
        if layer is None:
            return rys
        else:
            return rys[layer]
# step=11
# [ry.sum()for ry in rys]
# 24.6409-10.7561
# step=21
# 15.8421-10.5866
# step=31
# 11.8727-9.2988
# step=51
# 17.0698-14.4313
# 15.2912-12.8432





if __name__ == '__main__':
    model = get_model()
    filename = '../testImg.png'
    x=pilOpen(filename)
    x = pilToTensor(x).unsqueeze(0)
    x = toStd(x).to(device)
    hm=LIDIGDecomposer(model)(x, 243)
    print(hm)