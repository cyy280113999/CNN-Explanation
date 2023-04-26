"""
Integrated Gradient Decompose

by PathwayGradient
"""

import torch
import torch.nn.functional as nf
from utils import *


device = 'cuda'

# always over memory because cannot release hooks, when use class to object
def IGDecomposer(model, x, yc, x0="std0", layer_name=(None,), post_softmax=False, step=11, device=device):
    self = model
    self.hooks = []
    hookLayerByName(self, model, layer_name)
    # forward
    if x0 is None or x0 == "zero":
        x0 = torch.zeros_like(x)
    elif x0 == "std0":
        x0 = toStd(torch.zeros_like(x))
    else:
        raise Exception()
    xs = torch.zeros((step,) + x.shape[1:], device=device)
    xs[0] = x0[0]
    dx = (x - x0) / (step - 1)
    for i in range(1, step):
        xs[i] = xs[i-1] + dx
    xs.requires_grad_()
    output = model(xs)
    if post_softmax:
        output = nf.softmax(output, 1)
    o = output[:, yc].sum()  # all inputs will get its correct gradient
    o.backward()
    r = self.activation.diff(dim=0) * self.gradient[1:]
    r = r.sum([0, 1], True)

    # delete a&g reference
    self.activation = None
    self.gradient = None
    # delete graph reference
    model.zero_grad(set_to_none=True)
    for h in self.hooks:
        h.remove()
    return r



