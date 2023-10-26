import torch
import torch.nn.functional as nf
from utils import *

class SmoothTaylor:
    def __init__(self, model, layer_names):
        self.model = model
        self.layers = [findLayerByName(model, layer_name) for layer_name in layer_names]
        self.hooks = []

    def __call__(self, x, yc, n_samples=30, post_softmax=False):
        for layer in self.layers:
            self.hooks.append(layer.register_forward_hook(lambda *args, layer=layer: forward_hook(layer, *args))) # must capture by layer=layer
            self.hooks.append(layer.register_backward_hook(lambda *args, layer=layer: backward_hook(layer, *args)))
        gamma=0.1
        xs = x + gamma*torch.randn((n_samples,)+x.shape[1:],device=x.device)
        xs.requires_grad_()
        out = self.model(xs)
        if not post_softmax:
            out = nf.softmax(out, 1)
        out = out[0, yc]
        self.model.zero_grad()
        out.backward()
        hms = []
        with torch.no_grad():
            for layer in self.layers:
                hms.append((layer.activation * layer.gradient).sum(dim=[0, 1], keepdim=True))
            hm = multi_interpolate(hms)
        # clear hooks
        for layer in self.layers:
            layer.activation = None
            layer.gradient = None
        self.model.zero_grad(set_to_none=True)
        for h in self.hooks:
            h.remove()
        self.hooks = []
        return hm

if __name__ == '__main__':
    model = get_model()
    hmm = SmoothTaylor(model, [['features',30],['features',29]])
    x = get_image_x(filename='cat_dog_243_282.png', image_folder='../input_images/')
    x=x.cuda()
    hm=hmm(x,243)