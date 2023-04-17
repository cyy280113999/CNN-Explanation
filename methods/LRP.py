import matplotlib.pyplot as plt
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
        new_layer = torch.nn.Linear(layer.in_features, layer.out_features,
                                    bias=False, device=device)
    elif isinstance(layer, torch.nn.Conv2d):
        new_layer = torch.nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size,
                                    layer.stride, layer.padding, layer.dilation,
                                    layer.groups, padding_mode=layer.padding_mode,
                                    bias=False, device=device)
    else:
        return layer
    new_layer.weight = torch.nn.Parameter(fun(layer.weight))
    # new_layer.bias = torch.nn.Parameter() # if make same weight, remove bias for different
    return new_layer


def softmax_gradient(prob, target_class):
    t = torch.full_like(prob, -prob[0, target_class].item())
    t[0, target_class] += 1
    return t * prob


def softmax_int_grad(logits, yc, step=11):
    sig = torch.zeros_like(logits)
    dx = logits / (step - 1)  # closed interval
    lin_samples = torch.linspace(0, 1, step, device=device)
    # lines = []
    # fig=plt.figure()
    # axe=fig.add_subplot()
    for scale_multiplier in lin_samples:
        sig += softmax_gradient(nf.softmax(scale_multiplier * logits, dim=1), target_class=yc)
    sig *= dx
    #     lines.append(sig.cpu().clone().detach())
    # lines = torch.vstack(lines).numpy().T
    # axe.plot(lines)
    # axe.set_title(yc.item())
    return sig


def incr_AvoidNumericInstability(x, eps=1e-9):
    # -- zero often exists in conv layer
    # near_zero_count = (x.abs() <= eps).count_nonzero().item()
    # if near_zero_count > 0:
    #     print(f'instability neuron:{near_zero_count}')
    x = x + x.ge(0) * eps + x.lt(0) * (-eps)
    return x


def safeDivide(x, y, eps=1e-9):
    return (x / (y + (y.eq(0)) * eps)) * (y.ne(0))


def LRP_layer(layer, Ry, xs, layerMappings=None):
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
    if not isinstance(xs, (list, tuple)):
        xs = [xs]
    if not isinstance(layerMappings, (list, tuple)):
        layerMappings = [layerMappings]
    assert len(xs) == len(layerMappings)
    if isinstance(layer, (torch.nn.Linear, torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.Conv2d)):
        # lrp convert maxp to avgp
        MAXPOOL_REPLACE = False
        if MAXPOOL_REPLACE and isinstance(layer, torch.nn.MaxPool2d):
            layer = torch.nn.AvgPool2d(layer.kernel_size, layer.stride, layer.padding)
        # compatible for calculate
        # make sure each x has separate grad
        # clone: new memory
        # detach: new graph
        # not clone notice: do not modified the input tensor
        xs = [x.detach().requires_grad_(True) for x in xs]
        zs = []
        # forward
        for x, fun in zip(xs, layerMappings):
            if fun:
                new_layer = NewLayer(layer, fun)  # if given an identity function, creating new layer without bias.
                zs.append(new_layer.forward(x))
            else:
                zs.append(layer.forward(x))  # if given None , it has bias
        # -- new layer always remove bias. to keep bias, use map=None, or using map=identity to remove bias only.
        # BIAS = False
        # if BIAS:
        #     if hasattr(layer,'bias'):
        #         bias_dims = zs[0].shape
        #         bias_dims = [b if i < 2 else 1 for i, b in enumerate(bias_dims)]
        #         zs.append(layer.bias.view(bias_dims))
        z = sum(zs)
        # --1.avoid numeric instability by adding small value
        z = incr_AvoidNumericInstability(z)
        s = (Ry / z).detach()  # make sure s is constant, independent of the graph
        # --2.or in one step
        # s = safeDivide(Ry, z)
        # -- 1.this requires dimension matching
        # z.backward(s)
        # -- 2.this do not
        (z * s).sum().backward()
        if len(xs) == 1 and xs[0].shape[0] != 1:  # for IG
            Rx = xs[0][-1] * xs[0].grad.mean(0, True)
        else:
            Rs = [x * x.grad for x in xs]
            Rx = sum(Rs)
        # == jacobian version . j-mat is too big s.t. too slow.
        # x = x.clone().detach().requires_grad_(True)
        # y = layer.forward(x)
        # if method=='ydx':
        #     dy = y
        # elif method=='dydx':
        #     # dx = x-x0 = -sv
        #     x0 = x - dx
        #     dy = y-layer.forward(x0)
        # else:
        #     raise ValueError()
        # incr(dy)
        # dOdy = Ry/dy
        # dOdy_dims = dOdy.shape
        # dOdy_dims_idxs = list(range(len(dOdy_dims)))
        # dydx = torch.autograd.functional.jacobian(layer,x)
        # # dOdx = dOdy.mm(dydx)
        # for _ in dOdy_dims:
        #     dOdy = dOdy.unsqueeze(-1)
        # dOdx = (dOdy*dydx).sum(dOdy_dims_idxs,False)
        # Rx = dOdx * dx
        # Rx = Rx.sum(0, True)
        assert not Rx.isnan().any()
        return Rx.detach()
    elif isinstance(layer, (torch.nn.ReLU, torch.nn.Dropout)):
        return Ry
    elif isinstance(layer, (torch.nn.Flatten)):
        return Ry.reshape(xs[0][0].unsqueeze(0).shape)
    elif isinstance(layer, str) and layer == 'x layer':
        raise Exception()
    else:
        raise Exception()


def lrp0(activation):
    funs = None
    xs = activation.detach()
    return xs, funs


def lrpz(activation):
    funs = lambda w: w
    xs = activation.detach()
    return xs, funs


def lrpzp(activation):
    funs = lambda w: w.clip(min=0)
    xs = activation.detach()
    return xs, funs


def lrpw2(activation):
    funs = lambda w: w * w
    xs = torch.ones_like(activation)
    return xs, funs


def lrpgamma(activation, gamma=0.5):
    funs = lambda w: w + gamma * w.clip(min=0),
    xs = activation.detach()
    return xs, funs


def lrpzb_first(activation):
    funs = [
        lambda w: w,
        lambda w: w.clip(min=0),
        lambda w: w.clip(max=0),
    ]
    xs = [
        activation.detach(),
        -toStd(torch.zeros_like(activation)),
        -toStd(torch.ones_like(activation))
    ]
    return xs, funs


def lrpc(i, activation, flat_loc=None):
    # lrp composite
    # lrp0 -- lrp gamma -- lrp zb
    if not flat_loc:
        raise Exception('require flatten')
    if flat_loc + 1 <= i:
        return lrp0(activation)
    elif flat_loc == i:
        return lrp0(activation)
    elif 2 <= i <= flat_loc - 1:
        return lrpgamma(activation, 1.)
    elif 1 == i:
        return lrpzb_first(activation)
    else:
        raise ValueError('no layer')


def lrpig(activation, step=10):
    funs = None# lambda w: w.clip(min=0),
    xs = torch.zeros((step,) + activation.shape[1:], device='cuda')
    xs[0] = activation[0] / step  # notice: low value instability
    for i in range(1, step):
        xs[i] = xs[i - 1] + xs[1]
    return xs, funs


class LRP_Generator:
    def prepare_available_backward_init(self):
        self.available_layer_method = AvailableMethods({
            'lrp0',
            'lrpz',
            'lrpc',
            'lrpzp',
            'slrp',
            'lrpw2',
            'lrpig',  # new nonlinear propagation
        })
        self.available_backward_init = AvailableMethods({
            'yc',
            'normal',
            'target_one_hot',
            'c',
            'sg',
            'st',
            'sig0',
            'sigp',
        })
        # self.available_backward_init.__dict__.update({i: i for i in self.available_backward_init})
        # [self.available_backward_init.__setattr__(i, i) for i in self.available_backward_init]

    def __init__(self, model):
        self.prepare_available_backward_init()
        # layers is the used vgg
        self.model = model
        assert isinstance(model, (torchvision.models.VGG,
                                  torchvision.models.AlexNet))
        self.layers = ['input_layer'] + list(model.features) + [torch.nn.Flatten(1)] + list(model.classifier)
        self.flat_loc = 1 + len(list(model.features))
        self.layerlen = len(self.layers)

    def __call__(self, x, yc=None, backward_init='normal', method='lrpc', layer=None, device=device):
        # ___________runningCost___________= RunningCost(50)
        if layer:
            layer = auto_find_layer_index(self.model, layer)

        save_grad = True if method == self.available_layer_method.slrp else False

        # forward
        activations = [None] * self.layerlen
        x = x.to(device)
        if save_grad:
            x = x.requires_grad_()
        activations[0] = x
        for i in range(1, self.layerlen):
            activations[i] = self.layers[i](activations[i - 1])
            # store gradient
            if save_grad:
                activations[i].retain_grad()

        logits = activations[self.layerlen - 1]
        if yc is None:
            yc = logits.max(1)[1]
        elif isinstance(yc, int):
            yc = torch.LongTensor([yc]).detach().to(device)
        elif isinstance(yc, torch.Tensor):
            yc = yc.to(device)
        else:
            raise Exception()

        # Gradient backward if required
        target_onehot = nf.one_hot(yc, logits.shape[1]).float().detach()
        if save_grad:
            logits.backward(target_onehot)
        # ___________runningCost___________.tic()

        # LRP backward
        R = [None] * self.layerlen  # register memory
        if backward_init == self.available_backward_init.yc or \
                backward_init == self.available_backward_init.normal:
            R[self.layerlen - 1] = target_onehot * activations[self.layerlen - 1]
        elif backward_init == self.available_backward_init.target_one_hot:
            R[self.layerlen - 1] = target_onehot  # 1 else 0
        elif backward_init == self.available_backward_init.c:
            # (N-1)/N else -1/N
            R[self.layerlen - 1] = target_onehot + torch.full_like(logits, -1 / logits.shape[1])
        elif backward_init == self.available_backward_init.sg:  # softmax gradient
            R[self.layerlen - 1] = softmax_gradient(nf.softmax(logits, 1), yc)
        elif backward_init == self.available_backward_init.st:  # softmax taylor
            R[self.layerlen - 1] = activations[self.layerlen - 1] * softmax_gradient(nf.softmax(logits, 1), yc)
        elif backward_init == self.available_backward_init.sig0:  # softmax integrated gradient to 0
            sig = softmax_int_grad(logits, yc)
            R[self.layerlen - 1] = sig
        elif backward_init == self.available_backward_init.sigp:
            # Uncertain Calculation
            sig = torch.zeros_like(logits)
            sample_scale = 0.1
            dx = torch.zeros_like(logits)
            dx[0, yc] = logits[0, yc] * sample_scale
            lin_samples = sample_scale + torch.arange(0, 1, sample_scale, device=device)
            for scale_multiplier in lin_samples:
                lin_point = logits.clone()
                lin_point[0, yc] *= scale_multiplier
                prob = nf.softmax(lin_point, 1)
                prob_t = prob[0, yc]
                sgc = softmax_gradient(prob, yc)
                # sig += sgc * dx
                sig += sgc * sample_scale
            R[self.layerlen - 1] = sig
        else:
            raise Exception(f'Not Valid Method {backward_init}')
        # ___________runningCost___________.tic('last layer')
        _stop_at = layer if layer is not None else 0
        for i in range(_stop_at + 1, self.layerlen)[::-1]:
            if method == self.available_layer_method.lrp0:
                xs, funs = lrp0(activations[i - 1])
            elif method == self.available_layer_method.lrpz:
                xs, funs = lrpz(activations[i - 1])
            elif method == self.available_layer_method.lrpc:
                xs, funs = lrpc(i, activations[i - 1], flat_loc=self.flat_loc)
            elif method == self.available_layer_method.lrpzp:
                xs, funs = lrpzp(activations[i - 1])
            elif method == self.available_layer_method.slrp:
                xs, funs = lrpzp(activations[i - 1])
            elif method == self.available_layer_method.lrpw2:
                xs, funs = lrpw2(activations[i - 1])
            elif method == self.available_layer_method.lrpig:
                xs, funs = lrpig(activations[i - 1])
            else:
                raise Exception

            R[i - 1] = LRP_layer(self.layers[i], R[i], xs, funs)
            # ___________runningCost___________.tic(str(self.layers[i]))
            # bug when sglrp sum R == 0
            # R[i - 1] *= R[i].sum() / R[i - 1].sum()
            if method == self.available_layer_method.slrp \
                    and isinstance(self.layers[i - 1], torch.nn.Conv2d):
                grad = activations[i - 1].grad
                grad = grad.clip(min=0)
                grad = grad.sum([2, 3], True)
                grad /= grad.sum()
                # original_sum = R[i - 1].sum()
                R[i - 1] = R[i - 1] * grad
                # R[i - 1] *= original_sum / R[i - 1].sum()
        # ___________runningCost___________.cost()
        if layer is None:
            return R
        else:
            return R[layer]


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
        x = get_image_x(filename)
        for yc in ycs:
            R = lrpg(x, yc, method=method, sglrp=sglrp)
            # save_name = f'result/{filename}_cl{yc}{"_sglrp"if sglrp else""}{"_slrp"if slrp else""}.png'
            save_name = f'lrpvar/{filename}_cl{yc}_{method}{"_sglrp" if sglrp else ""}.png'
            relevance_gray_map = R[0].cpu().detach().sum(1, True)
            # save_plot(std_img=x.cpu().detach(), heatmap=relevance_gray_map, cmap='lrp',
            #           save_path=save_name)

    # mp=[0,5,10,17,24,31]
    # conv=[3,8,15,22,29]
    # save_name_layer = f'result/lrp{"_sglrp"if sglrp else""}{"_slrp"if slrp else""}{{}}.png'
    # for l in conv:
    #     relevance_gray_map = R[l].detach().sum(1,True)
    #     visualize(heatmap=relevance_gray_map, save_path=save_name_layer.format(l),cmap='lrp')
