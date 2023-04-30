"""
Integrated Gradient

"""

import torch
import torch.nn.functional as nf
from utils import *

device = 'cuda'

"""
if use class to define decomposer and use a lambda function to create a object,
it can not release hooks, always over memory.
to avoid that, there only gives a function definition.
"""


def IGDecomposer(model, x, yc, x0="std0", layer_name=(None,), post_softmax=False, step=11):
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
    xs = torch.zeros((step,) + x.shape[1:], device=x.device)
    xs[0] = x0[0]
    dx = (x - x0) / (step - 1)
    for i in range(1, step):
        xs[i] = xs[i - 1] + dx
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
    self.hooks = []
    return r
