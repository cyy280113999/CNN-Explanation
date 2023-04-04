'''
Part of code borrows from https://github.com/1Konny/gradcam_plus_plus-pytorch
'''

from ..cam import find_resnet_layer, find_densenet_layer, \
    find_squeezenet_layer, find_layer, find_googlenet_layer, find_mobilenet_layer, find_shufflenet_layer
import gc
from torchvision.models import VGG,AlexNet,ResNet,DenseNet,SqueezeNet,GoogLeNet,ShuffleNetV2,MobileNetV2,MobileNetV3
from utils import auto_find_layer_str


class BaseCAM(object):
    """ Base class for Class activation mapping.

        : Args
            - **model_dict -** : Dict. Has format as dict(type='vgg', arch=torchvision.models.vgg16(pretrained=True),
            layer_name='features',input_size=(224, 224)).

    """
    def __init__(self, model_dict):
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = None
        self.activations = None

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            return None

        def forward_hook(module, input, output):
            self.activations = output.detach()
            return None

        if isinstance(self.model_arch,(VGG,AlexNet)):
            self.target_layer = auto_find_layer_str(self.model_arch, layer_name)
        elif isinstance(self.model_arch, ResNet):
            self.target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif isinstance(self.model_arch, DenseNet):
            self.target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif isinstance(self.model_arch, SqueezeNet):
            self.target_layer = find_squeezenet_layer(self.model_arch, layer_name)
        elif isinstance(self.model_arch, GoogLeNet):
            self.target_layer = find_googlenet_layer(self.model_arch, layer_name)
        elif isinstance(self.model_arch, ShuffleNetV2):
            self.target_layer = find_shufflenet_layer(self.model_arch, layer_name)
        elif isinstance(self.model_arch, (MobileNetV2,MobileNetV3)):
            self.target_layer = find_mobilenet_layer(self.model_arch, layer_name)
        else:
            self.target_layer = find_layer(self.model_arch, layer_name)

        self.hooks=[]
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))


    def full_clear(self):
        # delete a&g reference
        self.activations = None
        self.gradients = None
        # delete graph reference
        self.model_arch.zero_grad(set_to_none=True)
        # collect unrefereed
        # gc.collect()  # this cost much time

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        print("clear hooks")

    # a cam object is not auto deleted. so hooks not released
    def __del__(self):
        self.clear_hooks()

