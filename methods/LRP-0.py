import torch
import torch.nn.functional as nf
from utils import *
device = 'cuda'

def incr_AvoidNumericInstability(x, eps=1e-9):
    return x + (x >= 0) * eps + (x < 0) * (-eps)
def prop_relev(layer, x, Ry):
    # if use this jacobian version, get CUDA out of memory.
    # x=x.squeeze(0)
    # Ry=Ry.squeeze(0)
    # x_deeps=len(x.shape)
    # y=layer.forward(x)
    # for i in range(x_deeps):
    #     y = y.unsqueeze(-1)
    #     Ry = Ry.unsqueeze(-1)
    # g=torch.autograd.functional.jacobian(lambda x:layer.forward(x),x)
    # # weight=g*y/incr_AvoidNumericInstability(x)
    # # r=Ry*weight
    # r=Ry*g*y/incr_AvoidNumericInstability(x)
    # Rx=r
    # for i in range(x_deeps):
    #     Rx=Rx.sum(0)

    # similar to grad prop
    x=x.clone().detach().requires_grad_()
    y = layer.forward(x)
    Gy = Ry / incr_AvoidNumericInstability(y)
    y.backward(Gy)
    Gx=x.grad
    Rx=x * Gx
    return Rx


def prop_grad(layer, x, Gy):
    x=x.clone().detach().requires_grad_()
    # if you give Ry, then Gy is Ry/y
    # when yi==0, so Ryi==0, it can not know real Gyi
    y = layer.forward(x)
    y.backward(Gy)
    Gx=x.grad
    # Do not return Rx=x*Gx, Gx instead.
    return Gx.clone().detach()



def LRP_0(model,x,yc,device=device):
    # create model
    layers = ['input layer'] + list(model.features) + [torch.nn.Flatten(1)] + list(model.classifier)

    # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
    # replace relu to non in-place
    for i,layer in enumerate(layers):
        if isinstance(layer,torch.nn.ReLU):
            layers[i]=torch.nn.ReLU(inplace=False)
    # flat_loc = len(list(model.features)) + 1
    layerlen = len(layers)

    # forward
    activations = [None] * layerlen
    x.requires_grad_().retain_grad()
    activations[0] = x
    for i in range(1, layerlen):
        activations[i] = layers[i](activations[i - 1])
    logits = activations[layerlen - 1]

    # backward
    yc = torch.LongTensor([yc]).detach().to(device)
    target_onehot = nf.one_hot(yc, logits.shape[1]).float().detach()
    Relevance_Propagate=False
    if Relevance_Propagate:
        R = [None] * layerlen
        R[layerlen - 1] = target_onehot * activations[layerlen - 1]
        for i in range(1, layerlen)[::-1]:
            R[i - 1] = prop_relev(layers[i],activations[i-1], R[i])
    else:# grad prop
        G = [None] * layerlen
        G[layerlen - 1] = target_onehot
        for i in range(1, layerlen)[::-1]:
            G[i - 1] = prop_grad(layers[i],activations[i-1], G[i])
        R=[a * g for a, g in zip(activations,G)]
    return R


if __name__ == '__main__':
    model = get_model()
    filename = '../testImg.png'
    x=pilOpen(filename)
    x = pilToTensor(x).unsqueeze(0)
    x = toStd(x).to(device)
    hm=LRP_0(model,x,243)
    print(hm)
