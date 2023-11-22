"""
Integrated Gradient

"""

import torch
import torch.nn.functional as nf
from utils import *

device = 'cuda'

"""
not same
    "IG-s5": lambda model: lambda x, y: interpolate_to_imgsize(
        IGDecomposer(model, x, y, layer_name=('features', 30), post_softmax=False)),
    "LID-IG-s5": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=5, bp=None, linear=False),
    "IG-s4": lambda model: lambda x, y: interpolate_to_imgsize(
        IGDecomposer(model, x, y, layer_name=('features', 23), post_softmax=False)),
    "LID-IG-s4": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=4, bp=None, linear=False),
    "IG-s3": lambda model: lambda x, y: interpolate_to_imgsize(
        IGDecomposer(model, x, y, layer_name=('features', 16), post_softmax=False)),
    "LID-IG-s3": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=3, bp=None, linear=False),
    "IG-s2": lambda model: lambda x, y: interpolate_to_imgsize(
        IGDecomposer(model, x, y, layer_name=('features', 9), post_softmax=False)),
    "LID-IG-s2": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=2, bp=None, linear=False),


if use class to define decomposer and use a lambda function to create a object,
it can not release hooks, always over memory.
to avoid that, there only gives a function definition.
"""
class IG:
    def __init__(self, model, layer_names, x0="std0", post_softmax=False, step=31, simplify=0, **kwargs):
        self.model = model
        self.layers = auto_hook(model, layer_names)
        self.x0=x0
        self.post_softmax=post_softmax
        self.step=step
        self.simplify=simplify

    def __call__(self, x, yc):
        # forward
        if self.x0 is None or self.x0 == "zero":
            x0 = torch.zeros_like(x)
        elif self.x0 == "std0":
            x0 = toStd(torch.zeros_like(x))
        else:
            raise Exception()
        xs = torch.zeros((self.step,) + x.shape[1:], device=x.device)
        xs[0] = x0[0]
        dx = (x - x0) / (self.step - 1)
        for i in range(1, self.step):
            xs[i] = xs[i - 1] + dx
        xs = xs.detach().requires_grad_()  # leaf node
        output = self.model(xs)
        if self.post_softmax:
            output = nf.softmax(output, 1)
        o = output[:, yc].sum()  # all inputs will get its correct gradient
        o.backward()
        hms=[]
        with torch.no_grad():
            for layer in self.layers:
                a = layer.activation.detach()
                g = layer.gradient
                if not self.simplify:
                    hm = a.diff(dim=0)*g[1:]
                else:
                    hm = (a[-1]-a[0]) * g.mean(0,True)
                hm = hm.sum([0, 1], True)
                hms.append(hm)
            hm = multi_interpolate(hms)
        return hm

    def __del__(self):
        # clear hooks
        for layer in self.layers:
            layer.activation = None
            layer.gradient = None
        self.model.zero_grad(set_to_none=True)
        clearHooks(self.model)
