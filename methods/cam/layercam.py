import torch
import torch.nn.functional as F
from ..cam.basecam import *
from utils import *


# --LayerCAM origin
# params: None
# --LayerCAM
# params: relu_weight=True, relu=True
class LayerCAM(BaseCAM):
    def __init__(self, model_dict):
        super().__init__(model_dict)

    def __call__(self, input, class_idx=None, retain_graph=False,
                sg=False, relu_weight=False, abs_=False,norm=False,relu=False):
        b, c, h, w = input.size()

        # predication on raw x
        logit = self.model_arch(input.cuda())
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
        else:
            predicted_class = torch.LongTensor([class_idx])

        if not sg:
            score = logit[0, predicted_class]
        else:
            prob = F.softmax(logit, 1)
            score = prob[0, predicted_class]
        self.model_arch.zero_grad()
        score.backward()

        activations = self.activations.detach()
        gradients = self.gradients.detach()

        with torch.no_grad():
            weights=gradients
            if relu_weight:
                weights=F.relu(weights)
            if abs_:
                weights=weights.abs()
            # if norm:
            #     weights
            cam = (activations * weights).sum(dim=1, keepdim=True)
            cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
            if relu:
                cam = F.relu(cam)
            cam = heatmapNormalizeR(cam)

        del self.activations
        del self.gradients
        return cam


