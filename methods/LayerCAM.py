import torch
import torch.nn.functional as nf
from utils import *


# -- LayerCAM: relu_weight=True, relu=True
# -- LayerCAM origin: None
class LayerCAM:
    def __init__(self, model, layer_names, relu_weight=True, relu=True,
                 post_softmax=False, abs_=False, norm=False, **kwargs):
        self.model = model
        self.layers = auto_hook(model, layer_names)
        self.relu_weight=relu_weight
        self.relu=relu
        self.post_softmax=post_softmax
        self.abs_=abs_
        self.norm=norm

    def __call__(self, x, yc=None):
        with torch.enable_grad():
            logit = self.model(x.requires_grad_())
            if yc is None:
                yc = logit.max(1)[-1]
            elif isinstance(yc, int):
                yc = torch.LongTensor([yc]).to(device)
            elif isinstance(yc, torch.Tensor):
                yc = yc.to(device)
            else:
                raise Exception()
            score = logit[0, yc]
            if self.post_softmax:
                prob = nf.softmax(logit, 1)
                score = prob[0, yc]
            self.model.zero_grad()
            score.backward()
        with torch.no_grad():
            hms = []
            for layer in self.layers:
                a = layer.activation.detach()
                g = layer.gradient
                weights = g
                if self.relu_weight:
                    weights = nf.relu(weights)
                if self.abs_:
                    weights = weights.abs()
                # if norm:
                #     weights
                cam = (a * weights).sum(dim=1, keepdim=True)
                # cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
                if self.relu:
                    cam = nf.relu(cam)
                # cam = heatmapNormalizeR(cam)
                hms.append(cam)
            cam = multi_interpolate(hms)
        return cam

    def __del__(self):
        # clear hooks
        for layer in self.layers:
            layer.activation = None
            layer.gradient = None
        self.model.zero_grad(set_to_none=True)
        clearHooks(self.model)

# if __name__ == '__main__':
#     model = get_model()
#     hmm = LayerCAM(model, [['features',30],['features',29]],sg=False, relu_weight=True, relu=True)
#     x = get_image_x(filename='cat_dog_243_282.png', image_folder='../input_images/')
#     x=x.cuda()
#     hm=hmm(x,243)