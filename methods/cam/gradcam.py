import torch
import torch.nn.functional as F
from utils import *
from ..cam.basecam import *



class GradCAM(BaseCAM):
    def __init__(self, model_dict):
        super().__init__(model_dict)

    def __call__(self, x, class_idx=None,
                 sg=False,
                 norm=False, abs_=False,
                 relu=True):
        x = x.cuda()
        b, c, h, w = x.size()

        # predication on raw x
        logit = self.model_arch(x)
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

        activation = self.activations.detach()
        gradient = self.gradients.detach()
        with torch.no_grad():
            weights = gradient.sum(dim=[2, 3], keepdim=True)
            if norm:
                weights = F.softmax(weights, 1)
            if abs_:
                weights=weights.abs()
            cam = (activation * weights).sum(dim=1, keepdim=True)
            cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
            if relu:
                cam = F.relu(cam)
                cam = heatmapNormalizeP(cam)
            else:
                cam = heatmapNormalizeR(cam)

        # with torch.no_grad():
        #     print(f'sg:{sg},cls:{predicted_class.item()}')
        #     print(f'score before {F.softmax(self.model_arch(x), 1)[0, predicted_class].item()}')
        #     print(
        #         f'score after {F.softmax(self.model_arch(x * binarize(cam)), 1)[0, predicted_class].item()}')

        self.full_clear()
        return cam

    def __del__(self):
        super().__del__()




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
