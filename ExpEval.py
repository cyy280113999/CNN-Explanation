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
from datasets.DiscrimDataset import DiscrimDataset,default_transform
from methods.AblationCAM import AblationCAM
from methods.cam.gradcam import GradCAM
from methods.cam.layercam import LayerCAM
from methods.LRP import LRP_Generator
from methods.IG import IGDecomposer
from methods.LIDLinearDecompose import LIDLinearDecomposer
from methods.LIDIGDecompose import LIDIGDecomposer
from methods.scorecam import ScoreCAM
from methods.RelevanceCAM import RelevanceCAM
from Evaluators.ProbChangeEvaluator import ProbChangeEvaluator
from Evaluators.MaximalPatchEvaluator import MaximalPatchEvaluator
from Evaluators.PointGameEvaluator import PointGameEvaluator


class EvaluatorSetter:
    def __init__(self):
        np.random.seed(1)
        torch.random.manual_seed(1)
        num_samples = 5000
        self.dataset_callers = {  # creating when called
            # ==imgnt val
            'sub_imgnt': lambda: [TD.Subset(ds, np.random.choice(len(ds), num_samples))
                                     for ds in [torchvision.datasets.ImageNet('F:/DataSet/imagenet', split='val',
                                                                              transform=default_transform)]][0],
            # ==discrim ds
            'DiscrimDataset': lambda: DiscrimDataset(transform=default_transform),
            # =relabeled imgnt
            'relabeled_top0': lambda: [TD.Subset(ds, np.random.choice(len(ds), num_samples)) for ds in [RI(topk=0)]][0],
            'relabeled_top1': lambda: [TD.Subset(ds, np.random.choice(len(ds), num_samples)) for ds in [RI(topk=1)]][0],
            # ==bbox imgnt
            'bbox_imgnt': lambda :[TD.Subset(ds, np.random.choice(len(ds), num_samples)) for ds in [BBImgnt()]][0],

        }

        self.models = {
            'vgg16': lambda: torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).eval().cuda()
        }

        # ---eval explaining methods
        cam_model_dict_by_layer = lambda model, layer: {'type': 'vgg16', 'arch': model, 'layer_name': f'{layer}',
                                                        'input_size': (224, 224)}
        interpolate_to_imgsize = lambda x: normalize_R(nf.interpolate(x.sum(1, True), 224, mode='bilinear'))
        multi_interpolate = lambda xs: normalize_R(
            sum(normalize_R(nf.interpolate(x.sum(1, True), 224, mode='bilinear')) for x in xs))
        self.heatmap_methods = {
            # base-line : cam, lrp top layer
            # "GradCAM-f": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, -1)).__call__,
            #                                    sg=False, relu=True),
            # "GradCAM-origin-f": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, '-1')).__call__,
            #                                           sg=False, relu=False),
            # "SG-GradCAM-origin-f": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, '-1')).__call__,
            #                                              sg=True, relu=False),
            # "LayerCAM-f": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '-1')).__call__,
            #                                            sg=False, relu_weight=True, relu=True),
            # "LRP-0-f": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpc', layer='-1')),
            # "SG-LRP-0-f": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer=-1)),

            # base-line : pixel layer
            # "SG-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer=1)),
            # "IG": lambda model: lambda x, y: interpolate_to_imgsize(
            #     IGDecomposer(model)(x, y)),

            # base-line : unimportant part
            # "ScoreCAM-f": lambda model: lambda x, y: ScoreCAM(model, '-1')(x, y, sg=True, relu=False),
            # "AblationCAM-f": lambda model: lambda x, y: AblationCAM(model, -1)(x, y, relu=False),
            # "RelevanceCAM-f": lambda model: lambda x, y: interpolate_to_imgsize(
            #     RelevanceCAM(model)(x, y, backward_init='c', method='lrpzp', layer=-1)),
            # "LRP-ZP-f": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpzp', layer=-1)),
            # "SG-LRP-ZP-f": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp', layer=-1)),

            # Increment Decomposition
            # "ST-LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer=-1)),
            # "SIG0-LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=-1)),

            # "LID-Taylor-f": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDLinearDecomposer(model)(x, y, layer=-1)),
            # "LID-Taylor-sig-f": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDLinearDecomposer(model)(x, y, layer=-1, backward_init='sig')),
            # "LID-IG-f": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDIGDecomposer(model)(x, y, layer=-1)),
            # "LID-IG-sig-f": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDIGDecomposer(model)(x, y, layer=-1, backward_init='sig')),

            # pixel level
            # "SIG0-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=1)),
            # "LID-Taylor-1": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDLinearDecomposer(model)(x, y, layer=1)),
            # "LID-Taylor-sig-1": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDLinearDecomposer(model)(x, y, layer=1, backward_init='sig')),
            # "LID-IG-1": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDIGDecomposer(model)(x, y, layer=1)),
            # "LID-IG-sig-1": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDIGDecomposer(model)(x, y, layer=1, backward_init='sig')),

            # differ layer
            "LID-IG-sig-24": lambda model: lambda x, y: interpolate_to_imgsize(
                LIDIGDecomposer(model)(x, y, layer=24, backward_init='sig')),
            "LID-IG-sig-17": lambda model: lambda x, y: interpolate_to_imgsize(
                LIDIGDecomposer(model)(x, y, layer=17, backward_init='sig')),
            "LID-IG-sig-10": lambda model: lambda x, y: interpolate_to_imgsize(
                LIDIGDecomposer(model)(x, y, layer=10, backward_init='sig')),
            "LID-IG-sig-5": lambda model: lambda x, y: interpolate_to_imgsize(
                LIDIGDecomposer(model)(x, y, layer=5, backward_init='sig')),
            # mix
            "SIG0-LRP-C-m": lambda model: lambda x, y: multi_interpolate(
                hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=None))
                if i in [1, 5, 10, 17, 24]),
            "LID-Taylor-sig-m": lambda model: lambda x, y: multi_interpolate(
                hm for i, hm in enumerate(LIDLinearDecomposer(model)(x, y, layer=None, backward_init='sig'))
                if i in [24, 31]),
            "LID-IG-sig-m": lambda model: lambda x, y: multi_interpolate(
                hm for i, hm in enumerate(LIDIGDecomposer(model)(x, y, layer=None, backward_init='sig'))
                if i in [1, 5, 10, 17, 24]),



            # "SIG0-LRP-C-24": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=24)),
            # "SIG0-LRP-C-17": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=17)),
            # "SIG0-LRP-C-10": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=10)),
            # "SIG0-LRP-C-5": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=5)),

            # step test
            # "LID-IG-sig-f-5": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDIGDecomposer(model)(x, y, layer=-1, backward_init='sig', step=5)),
            # "LID-IG-sig-f-11": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDIGDecomposer(model)(x, y, layer=-1, backward_init='sig', step=11)),
            # "LID-IG-sig-f-21": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDIGDecomposer(model)(x, y, layer=-1, backward_init='sig', step=21)),
            # "LID-IG-sig-f-31": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDIGDecomposer(model)(x, y, layer=-1, backward_init='sig', step=31)),
            # "LID-IG-sig-1-5": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDIGDecomposer(model)(x, y, layer=1, backward_init='sig', step=5)),
            # "LID-IG-sig-1-11": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDIGDecomposer(model)(x, y, layer=1, backward_init='sig', step=11)),
            # "LID-IG-sig-1-21": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDIGDecomposer(model)(x, y, layer=1, backward_init='sig', step=21)),
            # "LID-IG-sig-1-31": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDIGDecomposer(model)(x, y, layer=1, backward_init='sig', step=31)),

        }

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
    ds_name = 'DiscrimDataset'
    model_name = 'vgg16'
    mainEvaluator = EvaluatorSetter()
    EvalClass = MaximalPatchEvaluator
    mainEvaluator.presetting(ds_name, model_name)
    for hm_name in mainEvaluator.heatmap_methods:
        mainEvaluator.eval(hm_name, EvalClass)
