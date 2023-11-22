import torch
import torch.nn.functional as nf
from utils import *


class GradCAM:
    def __init__(self, model, layer_names,
                 relu=True,
                 post_softmax=False, norm=False, abs_=False, **kwargs):
        self.model = model
        self.layers = auto_hook(model, layer_names)
        self.relu=relu
        self.post_softmax=post_softmax
        self.norm=norm
        self.abs_=abs_

    def __call__(self, x, yc=None):
        logit = self.model(x.cuda())
        if yc is None:
            yc = logit.max(1)[-1]
        elif isinstance(yc, int):
            yc = torch.LongTensor([yc]).to(device)
        elif isinstance(yc, torch.Tensor):
            yc = yc.to(device)
        else:
            raise Exception()

        # origin version
        score = logit[0, yc]
        # new version , softmax gradient
        if self.post_softmax:
            score = nf.softmax(logit, 1)[0, yc]
        self.model.zero_grad()
        score.backward()
        with torch.no_grad():
            hms = []
            for layer in self.layers:
                a = layer.activation.detach()
                g = layer.gradient
                weights = g.sum(dim=[2, 3], keepdim=True)
                if self.norm:
                    weights = nf.softmax(weights, 1)
                if self.abs_:
                    weights = weights.abs()
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

# class GradCAM_test_memory_overflow:
#     def __init__(self, model, layer_name):
#         self.model_arch = model
#
#         self.gradients = None
#         self.activations = None
#
#         def backward_hook(module, grad_input, grad_output):
#             self.gradients = grad_output[0].clone().detach()
#             return None
#
#         def forward_hook(module, input, output):
#             self.activations = output.detach().clone().detach()
#             return None
#
#         self.target_layer = auto_find_layer(self.model_arch, layer_name)
#         self.target_layer.register_forward_hook(forward_hook)
#         self.target_layer.register_backward_hook(backward_hook)
#
#     def __call__(self, x, class_idx=None,
#                  sg=False,
#                  norm=False, abs_=False,
#                  relu=True):
#         x = x.cuda()
#         b, c, h, w = x.size()
#
#         # predication on raw x
#         logit = self.model_arch(x)
#         if class_idx is None:
#             predicted_class = logit.max(1)[-1]
#         else:
#             predicted_class = torch.LongTensor([class_idx])
#
#         # origin version
#         if not sg:
#             score = logit[0, predicted_class]
#         # new version , softmax gradient
#         else:
#             prob = F.softmax(logit, 1)
#             score = prob[0, predicted_class]
#
#         self.model_arch.zero_grad()
#         score.backward()
#
#         # activation = self.activations.detach()
#         # gradient = self.gradients.detach()
#
#         self.activations = None
#         self.gradients = None
#         self.model_arch.zero_grad(True)
#         gc.collect()



# ## delete activations to free memory
# def check_gradcam():
#     img = get_image(image_folder='../input_images/')
#     def one_process():
#         model = get_model()
#         gcam = GradCAM_test_memory_overflow(model,'-1')
#         gcam(img)
#         # torch.cuda.empty_cache()
#     for i in range(5):
#         one_process()
#     print('success')
# check_gradcam()
