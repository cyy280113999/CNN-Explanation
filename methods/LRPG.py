"""
LRP.py improved by grad propagation

theory of @cyy

"""
"""
lrp is stable
    "LRPG-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='normal', method='lrpc', layer_num=31)),
    "LRPG-ZP-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='normal', method='lrpzp', layer_num=31)),
    "LRPG-W2-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='normal', method='lrpw2', layer_num=31)),
    "LRPG-Gamma-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='normal', method='lrpgamma', layer_num=31)),
    "LRPG-AB-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='normal', method='lrpab', layer_num=31)),

c, sg, st is unstable because summation to zero
    "C-LRPG-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='c', method='lrpc', layer_num=31)),
    "SG-LRPG-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='sg', method='lrpc', layer_num=31)),
    "ST-LRPG-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='st', method='lrpc', layer_num=31)),

sig is stable
    "SIG-LRPG-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='sig', method='lrpc', layer_num=31)),
    "SIG-LRPG-C-24": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='sig', method='lrpc', layer_num=24)),
    "SIG-LRPG-C-17": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='sig', method='lrpc', layer_num=17)),
    "SIG-LRPG-C-10": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='sig', method='lrpc', layer_num=10)),
    "SIG-LRPG-C-5": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='sig', method='lrpc', layer_num=5)),
        
    "SIG-LRPG-ZP-24": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='sig', method='lrpzp', layer_num=24)),
    "SIG-LRPG-W2-24": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='sig', method='lrpw2', layer_num=24)),
    "SIG-LRPG-Gamma-24": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='sig', method='lrpgamma', layer_num=24)),
    "SIG-LRPG-AB-24": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='sig', method='lrpab', layer_num=24)),
    "SIG-LRPG-C-24": lambda model: lambda x, y: interpolate_to_imgsize(
        LRPWithGradient(model)(x, y, backward_init='sig', method='lrpc', layer_num=24)),

"""

import os
import torch
import torch.nn.functional as nf
from utils import *


def safeDivide(x, y, eps=1e-9):
    return (x / (y + y.ge(0) * eps + y.lt(0) * (-eps))) * (y.abs() > eps)


def softmax_gradient(prob, target_class):
    t = torch.full_like(prob, -prob[0, target_class].item())
    t[0, target_class] += 1
    return t * prob


def lrp_0_layer(layer, x, Gy):
    # using lrp-taylor method
    return lrp_taylor_layer(layer, x, Gy)
    # assert isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MaxPool2d,
    #                           torch.nn.AvgPool2d, torch.nn.AdaptiveAvgPool2d))
    # chain rule for linear:
    # w = layer.weight
    # Gx = Gy.mm(w)
    # for conv2d:
    # w = layer.weight
    # Gx = torch.conv_transpose2d(
    #     Gy,w,bias=None,stride=layer.stride,padding=layer.padding,
    #     groups=layer.groups,dilation=layer.dilation)
    # return Gx


def lrp_taylor_layer(layer, x, Gy):
    with torch.enable_grad():
        x = x.detach().requires_grad_()
        y = layer(x)
        # Gx = torch.autograd.grad(y, x, Gy)[0]
        (y * Gy).sum().backward()
    return x.grad


def lrp_zp_layer(layer, x, Gy):
    if isinstance(layer, torch.nn.Linear):
        w = layer.weight.clip(min=0)
        Gx = Gy.mm(w)
    elif isinstance(layer, torch.nn.Conv2d):
        w = layer.weight.clip(min=0)
        Gx = torch.conv_transpose2d(
            Gy, w, bias=None, stride=layer.stride, padding=layer.padding,
            groups=layer.groups, dilation=layer.dilation)
    else:
        raise Exception()
    return Gx


def lrp_w2_layer(layer, x, Gy):
    if isinstance(layer, torch.nn.Linear):
        w = layer.weight
        w = w * w
        Gx = Gy.mm(w)
    elif isinstance(layer, torch.nn.Conv2d):
        w = layer.weight
        w = w * w
        Gx = torch.conv_transpose2d(
            Gy, w, bias=None, stride=layer.stride, padding=layer.padding,
            groups=layer.groups, dilation=layer.dilation)
    else:
        raise Exception()
    return Gx


def lrp_gamma_layer(layer, x, Gy, gamma=0.5):
    if isinstance(layer, torch.nn.Linear):
        w = layer.weight
        w = w + gamma * w.clip(min=0)
        Gx = Gy.mm(w)
    elif isinstance(layer, torch.nn.Conv2d):
        w = layer.weight
        w = w + gamma * w.clip(min=0)
        Gx = torch.conv_transpose2d(
            Gy, w, bias=None, stride=layer.stride, padding=layer.padding,
            groups=layer.groups, dilation=layer.dilation)
    else:
        raise Exception()
    return Gx


def lrp_ab_layer(layer, x, Gy, a=1, b=1, auto_proportion=False):
    if auto_proportion:
        a, b = 1, 1  # that is lrp-0. so how to convert a,b between LRP and LRPG?
    if isinstance(layer, torch.nn.Linear):
        wp = layer.weight.clip(min=0)
        wn = layer.weight.clip(max=0)
        Gp = Gy.mm(wp)
        Gn = Gy.mm(wn)
        Gx = a * Gp + b * Gn
    elif isinstance(layer, torch.nn.Conv2d):
        wp = layer.weight.clip(min=0)
        wn = layer.weight.clip(max=0)
        Gp = torch.conv_transpose2d(
            Gy, wp, bias=None, stride=layer.stride, padding=layer.padding,
            groups=layer.groups, dilation=layer.dilation)
        Gn = torch.conv_transpose2d(
            Gy, wn, bias=None, stride=layer.stride, padding=layer.padding,
            groups=layer.groups, dilation=layer.dilation)
        Gx = a * Gp + b * Gn
    else:
        raise Exception()
    return Gx


# lrp-ig equals to lrp-0 for relu
def lrp_ig_layer(layer, x, Gy, step=10):
    xs = torch.zeros((step,) + x.shape[1:], device='cuda')
    xs[0] = x[0] / step  # no zero
    for i in range(1, step):
        xs[i] = xs[i - 1] + xs[0]
    with torch.enable_grad():
        xs = xs.detach().requires_grad_()
        y = layer(xs)
        (y * Gy).sum().backward()
    return xs.grad.mean(0, True).detach()


def lrp_zb_first(layer, x, Gy):
    assert isinstance(layer, torch.nn.Conv2d)
    l = toStd(torch.zeros_like(x))
    h = toStd(torch.ones_like(x))
    with torch.enable_grad():
        x = x.detach().requires_grad_()
        l.requires_grad_()
        h.requires_grad_()
        w = layer.weight
        wp = w.clip(min=0)
        wn = w.clip(max=0)
        yx = torch.conv2d(x, w, None, stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                          groups=layer.groups)
        yl = torch.conv2d(l, wp, None, stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                          groups=layer.groups)
        yh = torch.conv2d(h, wn, None, stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                          groups=layer.groups)
        y = yx - yl - yh
        (y * Gy).sum().backward()
    g = None
    r = x * x.grad - l * l.grad - h * h.grad
    return g, r


# LRP gradient propagation
class LRPWithGradient:
    def __init__(self, model):
        self.available_layer_method = AvailableMethods({
            'lrp0',
            # 'lrpz',  # in grad mode, lrpz=lrp0
            'lrpc',
            'lrpzp',
            'lrpw2',
            'lrpgamma',
            'lrpab',
            'lrpig',  # new nonlinear propagation
            'lrpc2',  # zp+zb
        })
        self.available_backward_init = AvailableMethods({
            'normal',
            'c',
            'sg',
            'st',
            'sig',
        })
        # layers is the used vgg
        self.model = model
        assert isinstance(model, (torchvision.models.VGG,
                                  torchvision.models.AlexNet))
        self.layers = ['input_layer'] + list(model.features) + [torch.nn.Flatten(1)] + list(model.classifier)
        self.flat_loc = 1 + len(list(model.features))
        self.layerlen = len(self.layers)

    def __call__(self, x, yc=None, backward_init='normal', method='lrpc', layer_num=None, device=device):
        if layer_num:
            layer_num = auto_find_layer_index(self.model, layer_num)
        # ============= forward
        with torch.no_grad():
            activations = [None] * self.layerlen
            x = x.to(device)
            activations[0] = x
            for i in range(1, self.layerlen):
                activations[i] = self.layers[i](activations[i - 1])
            logits = activations[self.layerlen - 1]
            if yc is None:
                yc = logits.max(1)[1]
            elif isinstance(yc, int):
                yc = torch.tensor([yc], device=device)
            elif isinstance(yc, torch.Tensor):
                yc = yc.to(device)
            else:
                raise Exception()
            target_onehot = nf.one_hot(yc, logits.shape[1]).float()

            # ============= LRP backward
            G = [None] * self.layerlen
            R = [None] * self.layerlen  # register memory
            if isinstance(backward_init, torch.Tensor):
                dody = backward_init  # ignore yc
            elif backward_init is None or backward_init == self.available_backward_init.normal:
                dody = target_onehot
            elif backward_init == "negative":
                dody = -target_onehot
            elif backward_init == self.available_backward_init.c:
                # (N-1)/N else -1/N
                R[-1] = target_onehot + torch.full_like(logits, -1 / logits.shape[1])
                G[-1] = safeDivide(R[-1], logits)
                assert not G[-1].isnan().any()
            elif backward_init == self.available_backward_init.sg:  # sg use softmax gradient as initial backward partial derivative
                R[-1] = softmax_gradient(nf.softmax(logits, 1), yc)
                G[-1] = safeDivide(R[-1], logits)
                assert not G[-1].isnan().any()
            elif backward_init == self.available_backward_init.st:  # softmax taylor
                G[-1] = softmax_gradient(nf.softmax(logits, 1), yc)
            elif backward_init == self.available_backward_init.sig:  # sig is a bad name as it definitely means average grad. Only refer to nonlinear initialization.
                step = 11
                dody = torch.zeros_like(logits)
                lin_samples = torch.linspace(0, 1, step, device=device)
                for scale_multiplier in lin_samples:
                    dody += softmax_gradient(nf.softmax(scale_multiplier * logits, 1), yc)
                dody /= step
            else:
                raise Exception(f'Not Valid Method {backward_init}')
            if G[-1] is None:
                G[-1] = dody.detach()
            if R[-1] is None:
                R[-1] = G[-1] * logits
            _stop_at = layer_num if layer_num is not None else 0
            for i in range(_stop_at + 1, self.layerlen)[::-1]:
                layer = self.layers[i]
                x = activations[i - 1]
                if isinstance(layer, torch.nn.Flatten):
                    G[i - 1] = G[i].reshape(x.shape)
                    R[i - 1] = R[i].reshape(x.shape)
                elif isinstance(layer, torch.nn.Dropout):
                    G[i - 1] = G[i]
                    R[i - 1] = R[i]
                elif isinstance(layer, torch.nn.MaxPool2d):
                    # lrp convert maxp to avgp
                    MAXPOOL_REPLACE = False
                    if MAXPOOL_REPLACE:
                        layer = torch.nn.AvgPool2d(layer.kernel_size, layer.stride, layer.padding)
                    G[i - 1] = lrp_taylor_layer(layer, x, G[i])
                    R[i - 1] = G[i - 1] * x
                elif isinstance(layer, torch.nn.ReLU):
                    """
                    Briefly, grad-lrp-relu clip grad to *(y>=0),
                    without lrp-relu , heatmap is always negatively.
                    so, all grad-lrp using relu(lrp-taylor).
                    
                    It is a interesting story about 
                    how grad-lrp0 with relu is the same as relev-lrp0 without relu.
                    It raise the question about the difference between grad prop and relev prop.
                    It reveals that relev-lrp is working for gradient recovering.
                    """
                    layer.inplace = False
                    G[i - 1] = lrp_taylor_layer(layer, x, G[i])
                    R[i - 1] = G[i - 1] * x
                elif isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
                    if method == self.available_layer_method.lrp0:
                        G[i - 1] = lrp_0_layer(layer, x, G[i])
                        R[i - 1] = G[i - 1] * x
                    elif method == self.available_layer_method.lrpzp:
                        G[i - 1] = lrp_zp_layer(layer, x, G[i])
                        R[i - 1] = G[i - 1] * x
                    elif method == self.available_layer_method.lrpw2:
                        G[i - 1] = lrp_w2_layer(layer, x, G[i])
                        R[i - 1] = G[i - 1] * x
                    elif method == self.available_layer_method.lrpgamma:
                        G[i - 1] = lrp_gamma_layer(layer, x, G[i])
                        R[i - 1] = G[i - 1] * x
                    elif method == self.available_layer_method.lrpab:
                        G[i - 1] = lrp_ab_layer(layer, x, G[i])
                        R[i - 1] = G[i - 1] * x
                    elif method == self.available_layer_method.lrpc:
                        # lrp composite, raised by @lrp-overview
                        # lrp0 -- lrp gamma -- lrp zb
                        if self.flat_loc + 1 <= i:
                            G[i - 1] = lrp_0_layer(layer, x, G[i])
                            R[i - 1] = G[i - 1] * x
                        elif 2 <= i <= self.flat_loc - 1:
                            G[i - 1] = lrp_gamma_layer(layer, x, G[i], 1)
                            R[i - 1] = G[i - 1] * x
                        elif 1 == i:
                            G[i - 1], R[i - 1] = lrp_zb_first(layer, x, G[i])
                        else:
                            raise ValueError('no layer')
                    elif method == self.available_layer_method.lrpig:
                        G[i - 1] = lrp_ig_layer(layer, x, G[i])
                        R[i - 1] = G[i - 1] * x
                    elif method == self.available_layer_method.lrpc2:
                        # lrp composite, raised by @CLRP
                        # lrp zp -- lrp zb
                        if 1 == i:
                            G[i - 1], R[i - 1] = lrp_zb_first(layer, x, G[i])
                        else:
                            G[i - 1] = lrp_zp_layer(layer, x, G[i])
                            R[i - 1] = G[i - 1] * x
                    else:
                        raise Exception(f'{method} is not available')
                elif isinstance(layer, str) and layer == 'input_layer':
                    raise Exception('finally end to input layer')
                else:
                    raise Exception(f'{layer} is not available')
        if layer_num is None:
            return R
        else:
            return R[layer_num]


if __name__ == '__main__':
    x = get_image_x()
    model = get_model('vgg16')
    d = LRPWithGradient(model)
    r = d(x, 243, 'sig', 'lrpc', 31).detach()
    show_heatmap(interpolate_to_imgsize(r))
    # print(r)
