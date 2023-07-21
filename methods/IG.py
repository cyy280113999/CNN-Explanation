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
    def __init__(self, model, layer_names):
        self.model = model
        self.layers = [findLayerByName(model, layer_name) for layer_name in layer_names]
    def __call__(self, x, yc, x0="std0", post_softmax=False, step=11):
        hooks=[]
        for layer in self.layers:
            hooks.append(layer.register_forward_hook(lambda *args, layer=layer: forward_hook(layer, *args))) # must capture by layer=layer
            hooks.append(layer.register_backward_hook(lambda *args, layer=layer: backward_hook(layer, *args)))
        # forward
        if x0 is None or x0 == "zero":
            x0 = torch.zeros_like(x)
        elif x0 == "std0":
            x0 = toStd(torch.zeros_like(x))
        else:
            raise Exception()
        xs = torch.zeros((step,) + x.shape[1:], device=x.device)
        xs[0] = x0[0]
        dx = (x - x0) / (step - 1)
        for i in range(1, step):
            xs[i] = xs[i - 1] + dx
        xs.requires_grad_()
        output = self.model(xs)
        if post_softmax:
            output = nf.softmax(output, 1)
        o = output[:, yc].sum()  # all inputs will get its correct gradient
        o.backward()
        with torch.no_grad():
            hms=[]
            for layer in self.layers:
                hm = layer.activation.diff(dim=0)*layer.gradient[1:]
                hm = hm.sum([0, 1], True)
                hms.append(hm)
            hm = multi_interpolate(hms)
        # clear hooks
        for layer in self.layers:
            layer.activation = None
            layer.gradient = None
        self.model.zero_grad(set_to_none=True)
        for h in hooks:
            h.remove()
        return hm
