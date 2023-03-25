"""
Integrated Gradient Decompose

by PathwayGradient
"""

import torch
import torch.nn.functional as nf
from utils import *
device='cuda'


class IGDecomposer:
    def __init__(self, model):
        self.model = model

    def __call__(self, x, yc, x0="std0", layer=None, post_softmax=False, step=11, device=device):
        # forward
        if x0 is None or x0=="zero":
            x0 = torch.zeros_like(x)
        elif x0=="std0":
            x0 = toStd(torch.zeros_like(x))
        else:
            raise Exception()
        xs = []
        delta_x = x - x0
        for i in range(0, step):
            xs.append(x0 + i/(step-1) * delta_x)
        xs = torch.vstack(xs).detach().requires_grad_()
        output=self.model(xs)
        if post_softmax:
            output=nf.softmax(output,1)
        o = output[:,yc].sum()  # all inputs will get its correct gradient
        o.backward()
        g = xs.grad.mean(0, True)  # avg grad to approximate IG
        return delta_x*g


