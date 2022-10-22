from PIL import Image
import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as nf
import itertools
import tqdm
from utils import *


def NewLayer(layer, fun=None):
    # copy a layer to gpu cost high time
    if not fun:
        return layer
    if isinstance(layer, torch.nn.Linear):
        new_layer = torch.nn.Linear(layer.in_features, layer.out_features, device=device)
    elif isinstance(layer, torch.nn.Conv2d):
        new_layer = torch.nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size,
                                    layer.stride, layer.padding, layer.dilation,
                                    layer.groups, padding_mode=layer.padding_mode, device=device)
    else:
        return layer
    try:
        new_layer.weight = torch.nn.Parameter(fun(layer.weight))
        new_layer.bias = torch.nn.Parameter(fun(layer.bias))
    except AttributeError:
        pass
    return new_layer


def LRP_layer(layer, Ry, xs, layerMappings=None):
    def incr(x, eps=1e-9):
        return x + (x == 0) * eps
    """
    @type xs: Union[List[tensor],tensor]
    @type layerMappings: Union[list[func],func]
    LRP process:
        x = x.detach().requires_grad(True)
        z = new_layer.forward(x)
        s = R / z
        z.backward(s)
        c = output[i].grad
        R = output[i] * c
    if more than one xs are given , calculate sum(Rs).
    """
    if isinstance(layer, (torch.nn.Linear, torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.Conv2d)) or (
            isinstance(layer, str) and layer == 'input layer'):
        # lrp convert maxp to avgp
        if isinstance(layer, torch.nn.MaxPool2d):
            layer = torch.nn.AvgPool2d(layer.kernel_size, layer.stride, layer.padding)
        # compatible for calculate
        if not isinstance(xs, list):
            xs = [xs]
        if not isinstance(layerMappings, list):
            layerMappings = [layerMappings]
        assert len(xs) == len(layerMappings)
        # make sure each x has separate grad
        xs = [x.clone().detach().requires_grad_(True) for x in xs]
        zs = []
        # forward
        for x, fun in zip(xs, layerMappings):
            if fun:
                layer = NewLayer(layer, fun)  # high time-consuming
            zs.append(layer.forward(x))
        z = sum(zs)
        # avoid numeric instability
        z = incr(z)
        s = (Ry / z)
        z.backward(s)
        Rs = [x * x.grad for x in xs]
        Rx = sum(Rs)
        return Rx

    elif isinstance(layer, (torch.nn.ReLU, torch.nn.Dropout)):
        return Ry
    elif isinstance(layer, (torch.nn.Flatten)):
        return Ry.reshape(xs[0].shape)


def lrpc(i, activation, flat_loc=None):
    # lrp composite
    # lrp0 -- lrp gamma -- lrp first
    if not flat_loc:
        raise Exception('require flatten')
    if flat_loc + 1 <= i:
        funs = [
            None
        ]
        xs = [
            activation
        ]

    elif flat_loc == i:
        funs = [
            None,
        ]
        xs = [
            activation
        ]

    elif 2 <= i <= flat_loc - 1:
        funs = [
            None,
            lambda x: 1. * x.clip(min=0),
        ]
        xs = [
            activation,
            activation  # copy
        ]

    elif 1 == i:
        funs = [
            lambda x: x,
            lambda x: -x.clip(min=0),
            lambda x: -x.clip(max=0),
        ]
        xs = [
            activation,
            transToStd(torch.zeros_like(activation)),
            transToStd(torch.ones_like(activation))
        ]

    else:
        raise ValueError('no layer')
    return xs, funs


def lrpz(i, activation, flat_loc=None):
    # lrpz is equivalent to taylor
    # Ry=y, y.grad==1, Ry == y*y.grad==taylor_y
    # when z=forward(x)==y
    # s=Ry/y ==taylor_y/y ==y.grad
    # z.backward(s) == z.backward(y.grad)
    # c=x.grad
    # Rx=x*x.grad==taylor_x
    funs = [
        None
    ]
    xs = [
        activation,
    ]
    return xs, funs


def lrpzp(i, activation, flat_loc=None):
    if i == 1:
        funs = [
            lambda x: x,
            lambda x: -x.clip(min=0),
            lambda x: -x.clip(max=0),
        ]
        xs = [
            activation,
            transToStd(torch.zeros_like(activation)),
            transToStd(torch.ones_like(activation))
        ]
    else:
        funs = [
            lambda x: x.clip(min=0),
        ]
        xs = [
            activation,
        ]
    return xs, funs


class LRP_Generator:
    def __init__(self, model):
        # layers is the used vgg
        assert isinstance(model, torchvision.models.VGG)
        self.layers = ['input layer'] + list(model.features) + [torch.nn.Flatten(1)] + list(model.classifier)
        self.flat_loc = 1 + len(list(model.features))
        self.layerlen = len(self.layers)

        pass

    def __call__(self, x, yc=None, slrp=False, backward_init='origin', method='lrpc', device=device):
        # ___________runningCost___________= RunningCost(50)
        # store output tensor beginning from input layer
        save_grad = True if slrp else False

        # forward
        activations = [None] * self.layerlen
        x = x.requires_grad_().to(device)
        activations[0] = x
        for i in range(1, self.layerlen):
            activations[i] = self.layers[i](activations[i - 1])
            # store gradient
            if save_grad:
                activations[i].retain_grad()

        logits = activations[self.layerlen - 1]
        if yc is None:
            yc = logits.max(1)[1]
        else:
            yc = torch.LongTensor([yc]).detach().to(device)

        # Gradient backward if required
        target_onehot = nf.one_hot(yc, logits.shape[1]).float().detach()
        if save_grad:
            logits.backward(target_onehot)
        # ___________runningCost___________.tic()

        # LRP backward
        R = [None] * self.layerlen  # register memory
        if backward_init == 'target_one_hot' or backward_init == 'origin' or backward_init == 'normal' or backward_init is None:
            # 1 else 0
            R[self.layerlen - 1] = target_onehot  # * activations[self.layerlen - 1]
        elif backward_init == 'clrp':
            # (N-1)/N else -1/N
            R[self.layerlen - 1] = target_onehot + torch.full_like(logits, -1 / logits.shape[1])
        elif backward_init == 'sglrp':
            # (1-p_t)p else -p_t*p
            prob = nf.softmax(logits, 1)
            prob_t = prob[0, yc]
            R[self.layerlen - 1] = (target_onehot - prob_t) * prob
        else:
            raise Exception(f'Not Valid Method {backward_init}')
        # ___________runningCost___________.tic('last layer')

        for i in range(1, self.layerlen)[::-1]:
            if method == 'lrpc':
                xs, funs = lrpc(i, activations[i - 1], flat_loc=self.flat_loc)
            elif method == 'lrpz':
                xs, funs = lrpz(i, activations[i - 1])
            elif method == 'lrpzp':
                xs, funs = lrpzp(i, activations[i - 1])

            assert not (R[i]).isnan().any()

            R[i - 1] = LRP_layer(self.layers[i], R[i], xs, funs)
            # ___________runningCost___________.tic(str(self.layers[i]))
            # bug when sglrp sum R == 0
            # R[i - 1] *= R[i].sum() / R[i - 1].sum()
            if slrp and isinstance(self.layers[i - 1], torch.nn.Conv2d):
                grad = activations[i - 1].grad
                grad = grad.sum([2, 3], True)
                grad *= (grad >= 0)
                grad /= grad.sum()
                original_sum = R[i - 1].sum()
                R[i - 1] = R[i - 1] * grad
                R[i - 1] *= original_sum / R[i - 1].sum()
        # ___________runningCost___________.cost()
        return R


auto_class = False
if __name__ == '__main__':
    model = get_model()
    ze_images = [['ze1_340_386.jpg', [340, 386]],
                 ['ze2.jpg', [340, 386]],
                 ['ze3.jpg', [340, 386]],
                 ['ze4.jpg', [340, 386]],
                 ['ze5.jpg', [340, 386]],
                 ]
    ca_images = [['castle_483_919_970.jpg', [483, 919, 970]],
                 ['cat_dog_243_282.png', [243, 282]],
                 ]
    eg_one = [['ILSVRC2012_val_00000518.JPEG', [336, 298, 272, 293, 191, ]]]
    images = eg_one
    sglrps = [False, True]
    slrps = [False, True]

    # for (filename,ycs),sglrp,slrp in tqdm.tqdm(itertools.product(images,sglrps,slrps)):
    #     if sglrp and slrp:
    #         continue
    #     x = get_image(filename)
    #     for yc in ycs:
    #         R = LRP(model,x,yc,sglrp,slrp)
    #         # save_name = f'result/{filename}_cl{yc}{"_sglrp"if sglrp else""}{"_slrp"if slrp else""}.png'
    #
    #         relevance_gray_map = R[0].cpu().detach().sum(1, True)
    #         visualize(std_img=x.cpu().detach(), heatmap=relevance_gray_map, cmap='lrp',
    #                   save_path=save_name)

    lrpvars = ['lrpc', 'lrpzp']
    lrpg = LRP_Generator(model)
    for (filename, ycs), method, sglrp in tqdm.tqdm(itertools.product(images, lrpvars, sglrps)):
        x = get_image(filename)
        for yc in ycs:
            R = lrpg(x, yc, method=method, sglrp=sglrp)
            # save_name = f'result/{filename}_cl{yc}{"_sglrp"if sglrp else""}{"_slrp"if slrp else""}.png'
            save_name = f'lrpvar/{filename}_cl{yc}_{method}{"_sglrp" if sglrp else ""}.png'
            relevance_gray_map = R[0].cpu().detach().sum(1, True)
            visualize(std_img=x.cpu().detach(), heatmap=relevance_gray_map, cmap='lrp',
                      save_path=save_name)

    # mp=[0,5,10,17,24,31]
    # conv=[3,8,15,22,29]
    # save_name_layer = f'result/lrp{"_sglrp"if sglrp else""}{"_slrp"if slrp else""}{{}}.png'
    # for l in conv:
    #     relevance_gray_map = R[l].detach().sum(1,True)
    #     visualize(heatmap=relevance_gray_map, save_path=save_name_layer.format(l),cmap='lrp')
