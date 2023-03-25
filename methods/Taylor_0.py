import torch.nn.functional as F
from utils import *

# ! hook not released!
# use layer=-1 as input layer
class Taylor_0:
    def __init__(self, model, layer_idx=30):
        def forward_hook(module, input, output):
            self.activations = output.clone().detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].clone().detach()
        def save_act_in(module, input, output):
            self.activations = input[0].clone().detach()
        def save_grad_in(module, grad_input, grad_output):
            self.gradients = grad_input[0].clone().detach()
        self.gradients = None
        self.activations = None

        self.model=model.cuda()
        features=list(model.features)
        if layer_idx==-1:
            # for input
            target_layer=features[0]
            target_layer.register_forward_hook(save_act_in)
            target_layer.register_backward_hook(save_grad_in)
        else:
            target_layer=features[layer_idx]
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_backward_hook(backward_hook)

    def __call__(self, input, class_idx):
        score = self.model(input.cuda().requires_grad_())[0, class_idx]
        self.model.zero_grad()
        score.backward()
        return self.activations * self.gradients
