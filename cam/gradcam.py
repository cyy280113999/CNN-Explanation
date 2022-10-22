import torch
import torch.nn.functional as F
import itertools
from utils import *
import torchvision.models as torchmodel
from cam.basecam import *



class GradCAM(BaseCAM):
    def __init__(self, model_dict):
        super().__init__(model_dict)

    def __call__(self, input, class_idx=None, retain_graph=False,
                 sg=False, norm=False, relu=True,abs_=False):
        b, c, h, w = input.size()

        # predication on raw input
        logit = self.model_arch(input.cuda())
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
        else:
            predicted_class = torch.LongTensor([class_idx])

        # origin version
        if not sg:
            score = logit[0, predicted_class]
        # new version , softmax gradient
        else:
            prob = F.softmax(logit, 1)
            score = prob[0, predicted_class]

        self.model_arch.zero_grad()
        score.backward()

        activations = self.activations.detach()
        gradients = self.gradients.detach()
        with torch.no_grad():
            weights = gradients.sum(dim=[2, 3], keepdim=True)
            if norm:
                weights = F.softmax(weights, 1)
            if abs_:
                weights=weights.abs()
            cam = (activations * weights).sum(dim=1, keepdim=True)
            cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
            if relu:
                cam = F.relu(cam)
                cam = normalize(cam)
            else:
                cam = normalize_R(cam)

        # with torch.no_grad():
        #     print(f'sg:{sg},cls:{predicted_class.item()}')
        #     print(f'score before {F.softmax(self.model_arch(input), 1)[0, predicted_class].item()}')
        #     print(
        #         f'score after {F.softmax(self.model_arch(input * binarize(cam)), 1)[0, predicted_class].item()}')

        return cam


