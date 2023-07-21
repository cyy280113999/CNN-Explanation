import torch
import torch.nn.functional as nf
from utils import *


# -- LayerCAM: relu_weight=True, relu=True
# -- LayerCAM origin: None
class LayerCAM:
    def __init__(self, model, layer_names):
        self.model = model
        self.layers = [findLayerByName(model, layer_name) for layer_name in layer_names]
        self.hooks = []

    def __call__(self, x, yc=None, relu_weight=True, relu=True,
                 post_softmax=False, abs_=False, norm=False):
        for layer in self.layers:
            self.hooks.append(layer.register_forward_hook(lambda *args, layer=layer: forward_hook(layer, *args))) # must capture by layer=layer
            self.hooks.append(layer.register_backward_hook(lambda *args, layer=layer: backward_hook(layer, *args)))
        logit = self.model(x.cuda())
        if yc is None:
            yc = logit.max(1)[-1]
        elif isinstance(yc, int):
            yc = torch.LongTensor([yc]).to(device)
        elif isinstance(yc, torch.Tensor):
            yc = yc.to(device)
        else:
            raise Exception()
        if not post_softmax:
            score = logit[0, yc]
        else:
            prob = nf.softmax(logit, 1)
            score = prob[0, yc]
        self.model.zero_grad()
        score.backward()

        with torch.no_grad():
            hms = []
            for layer in self.layers:
                weights = layer.gradient
                if relu_weight:
                    weights = nf.relu(weights)
                if abs_:
                    weights = weights.abs()
                # if norm:
                #     weights
                cam = (layer.activation * weights).sum(dim=1, keepdim=True)
                # cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
                if relu:
                    cam = nf.relu(cam)
                # cam = heatmapNormalizeR(cam)
                hms.append(cam)
        cam = multi_interpolate(hms)
        # clear hooks
        for layer in self.layers:
            layer.activation = None
            layer.gradient = None
        self.model.zero_grad(set_to_none=True)
        for h in self.hooks:
            h.remove()
        self.hooks = []
        return cam

if __name__ == '__main__':
    model = get_model()
    hmm = LayerCAM(model, [['features',30],['features',29]])
    x = get_image_x(filename='cat_dog_243_282.png', image_folder='../input_images/')
    x=x.cuda()
    hm=hmm(x,243,sg=False, relu_weight=True, relu=True)