"""
Taylor-0 equals to LRP-0

Taylor equals to DeepLIFT & LID-Taylor

"""
import torch
import torch.nn.functional as F
from utils import *


# "Taylor-30": lambda model:lambda x, y: interpolate_to_imgsize(Taylor_0(model, 30)(x, y)),

# ! hook not released!
class Taylor_0:
    def __init__(self, model, layer_name=(None,)):
        self.model = model
        self.layer = auto_hook(model, layer_name)[0]

    def __call__(self, x, yc):
        self.model.zero_grad()
        self.model(x.cuda().requires_grad_())[0, yc].backward()
        a = self.layer.activation
        g = self.layer.gradient
        return a * g

    def __del__(self):
        clearHooks(self.model)


class Taylor:
    def __init__(self, model, layer_name=(None,)):
        self.model = model
        self.layer = auto_hook(model, layer_name)[0]  # single layer

    def __call__(self, x, yc, x0='zero'):
        if x0 is None or x0 == 'zero':
            x0 = torch.zeros_like(x)
        elif x0 == 'std0':
            x0 = toStd(torch.zeros_like(x))
        else:
            raise Exception()
        x = torch.cat([x0,x])
        self.model.zero_grad()
        self.model(x.cuda().requires_grad_())[0, yc].backward()
        a = self.layer.activation
        g = self.layer.gradient
        return a.diff(dim=0) * g

    def __del__(self):
        clearHooks(self.model)