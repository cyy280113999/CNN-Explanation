import re
import json
from itertools import product
from functools import partial
import os

from tqdm import tqdm
import torch
import torch.utils.data as TD
import torch.nn.functional as nf
import torchvision
import numpy as np

# user
from utils import *
from datasets.bbox_imgnt import BBImgnt
from datasets.rrcri import RRCRI
from datasets.ri import RI
from datasets.DiscrimDataset import DiscrimDataset
from methods.AblationCAM import AblationCAM
from methods.cam.gradcam import GradCAM
from methods.cam.layercam import LayerCAM
from methods.LRP import LRP_Generator
from methods.IG import IGDecomposer
from methods.LIDLinearDecompose import LIDLinearDecomposer
from methods.LIDIGDecompose import LIDIGDecomposer
from methods.scorecam import ScoreCAM
from methods.RelevanceCAM import RelevanceCAM



class EvaluatorSetter:
    def __init__(self):
        np.random.seed(1)
        torch.random.manual_seed(1)
        num_samples = 5000
        get_indices = lambda dslen: np.random.choice(dslen, num_samples)
        self.dataset_callers = {  # creating when called
            # ==imgnt val
            'sub_imgnt': lambda: [TD.Subset(ds, get_indices(len(ds))) for ds in [getImageNet('val')]][0],
            # ==discrim ds
            'DiscrimDataset': lambda: DiscrimDataset(),
            # =relabeled imgnt
            'relabeled_top0': lambda: [TD.Subset(ds, get_indices(len(ds))) for ds in [RI(topk=0)]][0],
            'relabeled_top1': lambda: [TD.Subset(ds, get_indices(len(ds))) for ds in [RI(topk=1)]][0],
            # ==bbox imgnt
            'bbox_imgnt': lambda: [TD.Subset(ds, get_indices(len(ds))) for ds in [BBImgnt()]][0],

        }

        self.models = {
            'vgg16': lambda: get_model('vgg16'),
            'resnet34': lambda: get_model('resnet34'),
        }

        # ---eval explaining methods
        cam_model_dict_by_layer = lambda model, layer: {'type': 'vgg16', 'arch': model, 'layer_name': f'{layer}',
                                                        'input_size': (224, 224)}
        interpolate_to_imgsize = lambda x: heatmapNormalizeR(nf.interpolate(x.sum(1, True), 224, mode='bilinear'))
        multi_interpolate = lambda xs: heatmapNormalizeR(
            sum(heatmapNormalizeR(nf.interpolate(x.sum(1, True), 224, mode='bilinear')) for x in xs))
        from EvalSettings import eval_heatmap_methods
        self.heatmap_methods = eval_heatmap_methods

    def presetting(self,dataset_name, model_name):
        self.dataset_name = dataset_name
        self.dataset = self.dataset_callers[self.dataset_name]()
        self.dataloader = TD.DataLoader(self.dataset, batch_size=1, pin_memory=True, num_workers=2,
                                        persistent_workers=True)

        self.model_name = model_name
        self.model = self.models[self.model_name]()

    def eval(self, hm_name, SubEvalClass):
        self.heatmap_name = hm_name
        self.heatmap_method = self.heatmap_methods[self.heatmap_name](self.model)

        self.evaluator = SubEvalClass(self.dataset_name, self.dataset, self.dataloader,
                                      self.model_name, self.model,
                                      self.heatmap_name, self.heatmap_method)

        self.evaluator.eval()
        self.evaluator.save()


if __name__ == '__main__':
    print('utf8 chinese test: 中文测试')
    from EvalSettings import ds_name,model_name,EvalClass
    mainEvaluator = EvaluatorSetter()
    mainEvaluator.presetting(ds_name, model_name)
    for hm_name in mainEvaluator.heatmap_methods:
        mainEvaluator.eval(hm_name, EvalClass)
