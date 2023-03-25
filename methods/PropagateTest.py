import torch
import torch.nn.functional as nf
from torchvision.models import VGG, AlexNet

from utils import toStd
device='cuda'

def incr_AvoidNumericInstability(x, eps=1e-9):
    return x + (x >= 0) * eps + (x < 0) * (-eps)


BottomLayers = (
    torch.nn.Conv2d,
    torch.nn.MaxPool2d,
    torch.nn.AvgPool2d,
    torch.nn.Dropout,
    torch.nn.BatchNorm2d,
    torch.nn.AdaptiveAvgPool2d,
    torch.nn.Flatten,
    torch.nn.Linear,
    torch.nn.Softmax,
)
InplaceLayers=(
    torch.nn.ReLU,
)


def meanGradProp(x, x0, layer, Gy, step=2):
    if isinstance(layer, BottomLayers):
        xs = torch.zeros((step,)+layer.x.shape[1:], device='cuda')
        xs[0] = layer.x[0]
        dx = (layer.x[1] - layer.x[0]) / (step - 1)
        for i in range(1, step):
            xs[i] = xs[i - 1] + dx
        xs.requires_grad_()
        ys = layer(xs)
        (ys * Gy).sum().backward()
        Gx=xs.grad.mean(0, True).detach()
        with torch.no_grad():
            layer.Rx=Gx*dx
        return Gx
    elif isinstance(layer, torch.nn.Sequential):
        xs=[x]
        x0s=[x0]
        for m in layer.modules:
            xs.append(m(xs[-1]))
            x0s.append(m(x0s[-1]))
        for i, m in enumerate(layer.modules)[::-1]:
            Gy = meanGradProp(xs[i],x0s[i],m,Gy,step=step)
            with torch.no_grad():
                layer.Rx=(xs[i]-x0s[i])*Gy
        return Gy
    elif isinstance(layer,(VGG,AlexNet)):
        f=layer.features(x)
        f0=layer.features(x0)
        Gy=meanGradProp(f,f0,layer.classifier,Gy,step=step)
        Gy.reshape(f.shape)
        Gy=meanGradProp(x,x0,layer.features,Gy,step=step)
        return Gy


class LIDDecomposer:
    def __init__(self, model):
        self.model = model
        self.hooks = []

    def __call__(self, x, yc, x0="std0", layer=None, backward_init="normal", step=21, device=device):
        if x0 is None or x0 == "zero":
            x0 = torch.zeros_like(x)
        elif x0 == "std0":
            x0 = toStd(torch.zeros_like(x))
        else:
            raise Exception()
        x=torch.vstack([x0,x])
        if isinstance(yc, torch.Tensor):
            yc = yc.item()
        if isinstance(backward_init, torch.Tensor):
            dody = backward_init  # ignore yc
        elif backward_init is None or backward_init == "normal":
            dody = nf.one_hot(torch.tensor([yc], device=device), (1,1000))
        else:
            raise Exception()
        Gy=meanGradProp(x,self.model,dody,step=step)