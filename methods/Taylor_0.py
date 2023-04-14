import torch.nn.functional as F
from utils import *


# ! hook not released!
# use layer=-1 as input layer
class Taylor_0:
    def __init__(self, model, layer_name=(None,)):
        self.model = model
        self.hooks = []
        hookLayerByName(self, model, layer_name)

    def __call__(self, input, class_idx):
        score = self.model(input.cuda().requires_grad_())[0, class_idx]
        self.model.zero_grad()
        score.backward()
        return self.activations * self.gradients
