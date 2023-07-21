import torch.utils.data as TD
from datasets.BBoxImgnt import BBImgnt
from datasets.rrcri import RRCRI
from datasets.ri import RI
from datasets.ImgntTop5 import ImgntTop5
from datasets.DiscrimDataset import DiscrimDataset
from Evaluators.ProbChangeEvaluator import ProbChangeEvaluator
from Evaluators.MaximalPatchEvaluator import MaximalPatchEvaluator
from Evaluators.PointGameEvaluator import PointGameEvaluator
from HeatmapMethods import *


def rand_choice_ds(ds):
    import time
    import numpy as np
    np.random.seed(1)  # this is repeatable
    num_samples = 5000
    dslen = len(ds)
    indices = np.random.choice(dslen, num_samples)
    ds = TD.Subset(ds, indices)
    np.random.seed(int(time.time())) # recover seed
    return ds


dataset_callers = {  # creating when called
    # ==imgnt val
    'sub_imgnt': lambda: rand_choice_ds(getImageNet('val')),
    'sub_top5': lambda: rand_choice_ds(ImgntTop5(model_type='vgg16')),
    # ==discrim ds
    'DiscrimDataset': lambda: DiscrimDataset(),
    # =relabeled imgnt
    'relabeled_top0': lambda: rand_choice_ds(RI(topk=0)),
    'relabeled_top1': lambda: rand_choice_ds(RI(topk=1)),
    # ==bbox imgnt
    'bbox_imgnt': lambda: rand_choice_ds(BBImgnt()),
}
models = {
    'vgg16': lambda: get_model('vgg16'),
    'resnet34': lambda: get_model('resnet34'),
    'googlenet': lambda: get_model('googlenet'),
}
# settings
ds_name = 'sub_top5'
model_name = 'vgg16'
EvalClass = MaximalPatchEvaluator

eval_vis_check = False
eval_heatmap_methods = {
    # base-line : cam, lrp top layer
    # "GradCAM-s5": lambda model: partial(GradCAM(model, decode_stages(model, 5)).__call__, post_softmax=False, relu=True),
    # "GradCAM-origin-s5": lambda model: partial(GradCAM(model, decode_stages(model, 5)).__call__, post_softmax=False, relu=False),
    # "SG-GradCAM-origin-s5": lambda model: partial(GradCAM(model, decode_stages(model, 5)).__call__, post_softmax=True, relu=False),
    # "LayerCAM-s5": lambda model: partial(LayerCAM(model, decode_stages(model, 5)).__call__, post_softmax=False, relu_weight=True, relu=True),
    # "ScoreCAM-s5": lambda model: partial(ScoreCAM(model, decode_stages(model, 5)).__call__, post_softmax=True, relu=False),

    # "LRP-0-s5": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrp0', layer_num=31)),
    # "LRP-ZP-s5": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpzp', layer_num=31)),
    # "SG-LRP-0-s5": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrp0', layer_num=-1)),
    # "SG-LRP-ZP-s5": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp', layer_num=31)),
    # "IG-s5": lambda model: lambda x, y: interpolate_to_imgsize(
    #     IGDecomposer(model, x, y, layer_name=('features', -1), post_softmax=False)),
    # "SIG-s5": lambda model: lambda x, y: interpolate_to_imgsize(
    #     IGDecomposer(model, x, y, layer_name=('features', -1), post_softmax=True)),

    # base-line : unimportant part
    # "Random": lambda model: lambda x,y: normalize_R(torch.randn((1,)+x.shape[-2:])),
    # "AblationCAM-s5": lambda model: lambda x, y: AblationCAM(model, -1)(x, y, relu=False),
    # "RelevanceCAM-s5": lambda model: lambda x, y: interpolate_to_imgsize(
    #     RelevanceCAM(model)(x, y, backward_init='c', method='lrpzp', layer=-1)),

    # improvement
    # "ST-LRP-0-s5": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrp0', layer_num=-1)),
    # "ST-LRP-ZP-s5": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpzp', layer_num=31)),
    # "SIG-LRP-0-s5": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig', method='lrpc', layer_num=-1)),
    # "SIG-LRP-ZP-s5": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig', method='lrpzp', layer_num=-1)),


    # Increment Decomposition
    # "LID-Taylor-s5": lambda model: lambda x, y: LID_m_caller(model, x, y, s=5, bp=None, lin=True),
    # "LID-IG-s5": lambda model: lambda x, y: LID_m_caller(model, x, y, s=5, bp=None, lin=False),
    # "LID-Taylor-s5-r0": lambda model: lambda x, y: LID_m_caller(model, x, y, x0='zero', s=5, bp=None, lin=True), # refer to zero, not many changes
    # "LID-IG-s5-r0": lambda model: lambda x, y: LID_m_caller(model, x, y, x0='zero', s=5, bp=None, lin=False),
    # "SIG-LID-Taylor-s5": lambda model: lambda x, y: LID_m_caller(model, x, y, s=5, bp='sig', lin=True),
    # "SIG-LID-IG-s5": lambda model: lambda x, y: LID_m_caller(model, x, y, s=5, bp='sig', lin=False),
    # "SIG-LID-Taylor-s5-r0": lambda model: lambda x, y: LID_m_caller(model, x, y, x0='zero', s=5, bp='sig', lin=True),
    # "SIG-LID-IG-s5-r0": lambda model: lambda x, y: LID_m_caller(model, x, y, x0='zero', s=5, bp='sig', lin=False),

    # ========= middle layer
    # "LRP-0-s4": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init=None, method='lrp0', layer_num=24)),  # for vgg16
    # "LRP-0-s3": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init=None, method='lrp0', layer_num=17)),
    # "LRP-0-s2": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init=None, method='lrp0', layer_num=10)),
    # "LRP-0-s1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init=None, method='lrp0', layer_num=5)),
    # "LRP-C-s4": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init=None, method='lrpc', layer_num=24)), # for vgg16
    # "LRP-C-s3": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init=None, method='lrpc', layer_num=17)),
    # "LRP-C-s2": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init=None, method='lrpc', layer_num=10)),
    # "LRP-C-s1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init=None, method='lrpc', layer_num=5)),
    # "LRP-ZP-s4": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init=None, method='lrpzp', layer_num=24)),
    # "LRP-ZP-s3": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init=None, method='lrpzp', layer_num=17)),
    # "LRP-ZP-s2": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init=None, method='lrpzp', layer_num=10)),
    # "LRP-ZP-s1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init=None, method='lrpzp', layer_num=5)),
    # "IG-s4": lambda model: partial(IG(model, decode_stages(model, 4)).__call__, post_softmax=False),
    # "IG-s3": lambda model: partial(IG(model, decode_stages(model, 3)).__call__, post_softmax=False),
    # "IG-s2": lambda model: partial(IG(model, decode_stages(model, 2)).__call__, post_softmax=False),
    # "IG-s1": lambda model: partial(IG(model, decode_stages(model, 1)).__call__, post_softmax=False),
    # "SIG-s4": lambda model: partial(IG(model, decode_stages(model, 4)).__call__, post_softmax=True),
    # "SIG-s3": lambda model: partial(IG(model, decode_stages(model, 3)).__call__, post_softmax=True),
    # "SIG-s2": lambda model: partial(IG(model, decode_stages(model, 2)).__call__, post_softmax=True),
    # "SIG-s1": lambda model: partial(IG(model, decode_stages(model, 1)).__call__, post_softmax=True),
    # "SIG-LRP-C-s4": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig', method='lrpc', layer_num=24)),
    # "SIG-LRP-C-s3": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig', method='lrpc', layer_num=17)),
    # "SIG-LRP-C-s2": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig', method='lrpc', layer_num=10)),
    # "SIG-LRP-C-s1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig', method='lrpc', layer_num=5)),
    # "SIG-LID-Taylor-s4": lambda model: lambda x, y: LID_m_caller(model, x, y, s=4, bp='sig', lin=True),
    # "SIG-LID-Taylor-s3": lambda model: lambda x, y: LID_m_caller(model, x, y, s=3, bp='sig', lin=True),
    # "SIG-LID-Taylor-s2": lambda model: lambda x, y: LID_m_caller(model, x, y, s=2, bp='sig', lin=True),
    # "SIG-LID-Taylor-s1": lambda model: lambda x, y: LID_m_caller(model, x, y, s=1, bp='sig', lin=True),
    # "SIG-LID-IG-s4": lambda model: lambda x, y: LID_m_caller(model, x, y, s=4, bp='sig', lin=False),
    # "SIG-LID-IG-s3": lambda model: lambda x, y: LID_m_caller(model, x, y, s=3, bp='sig', lin=False),
    # "SIG-LID-IG-s2": lambda model: lambda x, y: LID_m_caller(model, x, y, s=2, bp='sig', lin=False),
    # "SIG-LID-IG-s1": lambda model: lambda x, y: LID_m_caller(model, x, y, s=1, bp='sig', lin=False),
    # "SIG-LID-Taylor-CE10-s4": lambda model: lambda x, y: LID_m_caller(model, x, y, s=4, bp='sig', lin=1, ce=1.0),
    # "SIG-LID-Taylor-CE10-s3": lambda model: lambda x, y: LID_m_caller(model, x, y, s=3, bp='sig', lin=1, ce=1.0),
    # "SIG-LID-Taylor-CE10-s2": lambda model: lambda x, y: LID_m_caller(model, x, y, s=2, bp='sig', lin=1, ce=1.0),
    # "SIG-LID-Taylor-CE10-s1": lambda model: lambda x, y: LID_m_caller(model, x, y, s=1, bp='sig', lin=1, ce=1.0),
    # "SIG-LID-Taylor-CE05-s4": lambda model: lambda x, y: LID_m_caller(model, x, y, s=4, bp='sig', lin=1, ce=0.5),
    # "SIG-LID-Taylor-CE05-s3": lambda model: lambda x, y: LID_m_caller(model, x, y, s=3, bp='sig', lin=1, ce=0.5),
    # "SIG-LID-Taylor-CE05-s2": lambda model: lambda x, y: LID_m_caller(model, x, y, s=2, bp='sig', lin=1, ce=0.5),
    # "SIG-LID-Taylor-CE05-s1": lambda model: lambda x, y: LID_m_caller(model, x, y, s=1, bp='sig', lin=1, ce=0.5),
    # "SIG-LID-Taylor-CE02-s4": lambda model: lambda x, y: LID_m_caller(model, x, y, s=4, bp='sig', lin=1, ce=0.2),
    # "SIG-LID-Taylor-CE02-s3": lambda model: lambda x, y: LID_m_caller(model, x, y, s=3, bp='sig', lin=1, ce=0.2),
    # "SIG-LID-Taylor-CE02-s2": lambda model: lambda x, y: LID_m_caller(model, x, y, s=2, bp='sig', lin=1, ce=0.2),
    # "SIG-LID-Taylor-CE02-s1": lambda model: lambda x, y: LID_m_caller(model, x, y, s=1, bp='sig', lin=1, ce=0.2),
    # "SIG-LID-IG-CE10-s4": lambda model: lambda x, y: LID_m_caller(model, x, y, s=4, bp='sig', lin=0, ce=1.0),
    # "SIG-LID-IG-CE10-s3": lambda model: lambda x, y: LID_m_caller(model, x, y, s=3, bp='sig', lin=0, ce=1.0),
    # "SIG-LID-IG-CE10-s2": lambda model: lambda x, y: LID_m_caller(model, x, y, s=2, bp='sig', lin=0, ce=1.0),
    # "SIG-LID-IG-CE10-s1": lambda model: lambda x, y: LID_m_caller(model, x, y, s=1, bp='sig', lin=0, ce=1.0),
    # "SIG-LID-IG-CE05-s4": lambda model: lambda x, y: LID_m_caller(model, x, y, s=4, bp='sig', lin=0, ce=0.5),
    # "SIG-LID-IG-CE05-s3": lambda model: lambda x, y: LID_m_caller(model, x, y, s=3, bp='sig', lin=0, ce=0.5),
    # "SIG-LID-IG-CE05-s2": lambda model: lambda x, y: LID_m_caller(model, x, y, s=2, bp='sig', lin=0, ce=0.5),
    # "SIG-LID-IG-CE05-s1": lambda model: lambda x, y: LID_m_caller(model, x, y, s=1, bp='sig', lin=0, ce=0.5),
    # "SIG-LID-IG-CE02-s4": lambda model: lambda x, y: LID_m_caller(model, x, y, s=4, bp='sig', lin=0, ce=0.2),
    # "SIG-LID-IG-CE02-s3": lambda model: lambda x, y: LID_m_caller(model, x, y, s=3, bp='sig', lin=0, ce=0.2),
    # "SIG-LID-IG-CE02-s2": lambda model: lambda x, y: LID_m_caller(model, x, y, s=2, bp='sig', lin=0, ce=0.2),
    # "SIG-LID-IG-CE02-s1": lambda model: lambda x, y: LID_m_caller(model, x, y, s=1, bp='sig', lin=0, ce=0.2),
    # =============  pixel level ============
    # "SG-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer=1)),
    # "SG-LRP-ZP-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp', layer=1)),
    # "IG": lambda model: lambda x, y: interpolate_to_imgsize(
    #     IGDecomposer(model)(x, y)),
    # "ST-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer=1)),
    # "SIG-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig', method='lrpc', layer=1)),

    # ================  mix layer ===================
    # "LayerCAM-s54": lambda model: partial(LayerCAM(model, decode_stages(model, (5, 4))).__call__, post_softmax=False, relu_weight=True, relu=True),
    # "LayerCAM-s543": lambda model: partial(LayerCAM(model, decode_stages(model, (5, 4, 3))).__call__, post_softmax=False, relu_weight=True, relu=True),
    # "LayerCAM-s5432": lambda model: partial(LayerCAM(model, decode_stages(model, (5, 4, 3, 2))).__call__, post_softmax=False, relu_weight=True, relu=True),
    # "LayerCAM-s54321": lambda model: partial(LayerCAM(model, decode_stages(model, (5, 4, 3, 2, 1))).__call__, post_softmax=False, relu_weight=True, relu=True),
    # "SG-LRP-0-s54": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sg', method='lrp0', layer_num=None)) if
    #     i in [24, 31]),
    # "SG-LRP-0-s543": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sg', method='lrp0', layer_num=None)) if
    #     i in [17, 24, 31]),
    # "SG-LRP-0-s5432": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sg', method='lrp0', layer_num=None)) if
    #     i in [10, 17, 24, 31]),
    # "SG-LRP-0-s54321": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sg', method='lrp0', layer_num=None)) if
    #     i in [5, 10, 17, 24, 31]),
    # "SG-LRP-ZP-s54": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp', layer_num=None)) if
    #     i in [24, 31]),
    # "SG-LRP-ZP-s543": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp', layer_num=None)) if
    #     i in [17, 24, 31]),
    # "SG-LRP-ZP-s5432": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp', layer_num=None)) if
    #     i in [10, 17, 24, 31]),
    # "SG-LRP-ZP-s54321": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp', layer_num=None)) if
    #     i in [5, 10, 17, 24, 31]),
    "SG-LRP-C-s54": lambda model: lambda x, y: multi_interpolate(
        hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer_num=None)) if
        i in [24, 31]),
    "SG-LRP-C-s543": lambda model: lambda x, y: multi_interpolate(
        hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer_num=None)) if
        i in [17, 24, 31]),
    "SG-LRP-C-s5432": lambda model: lambda x, y: multi_interpolate(
        hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer_num=None)) if
        i in [10, 17, 24, 31]),
    "SG-LRP-C-s54321": lambda model: lambda x, y: multi_interpolate(
        hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer_num=None)) if
        i in [5, 10, 17, 24, 31]),
    # "ST-LRP-0-s54": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='st', method='lrp0', layer_num=None)) if
    #     i in [24, 31]),
    # "ST-LRP-0-s543": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='st', method='lrp0', layer_num=None)) if
    #     i in [17, 24, 31]),
    # "ST-LRP-0-s5432": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='st', method='lrp0', layer_num=None)) if
    #     i in [10, 17, 24, 31]),
    # "ST-LRP-0-s54321": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='st', method='lrp0', layer_num=None)) if
    #     i in [5, 10, 17, 24, 31]),
    # "ST-LRP-ZP-s54": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='st', method='lrpzp', layer_num=None)) if
    #     i in [24, 31]),
    # "ST-LRP-ZP-s543": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='st', method='lrpzp', layer_num=None)) if
    #     i in [17, 24, 31]),
    # "ST-LRP-ZP-s5432": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='st', method='lrpzp', layer_num=None)) if
    #     i in [10, 17, 24, 31]),
    # "ST-LRP-ZP-s54321": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='st', method='lrpzp', layer_num=None)) if
    #     i in [5, 10, 17, 24, 31]),
    "ST-LRP-C-s54": lambda model: lambda x, y: multi_interpolate(
        hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer_num=None)) if
        i in [24, 31]),
    "ST-LRP-C-s543": lambda model: lambda x, y: multi_interpolate(
        hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer_num=None)) if
        i in [17, 24, 31]),
    "ST-LRP-C-s5432": lambda model: lambda x, y: multi_interpolate(
        hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer_num=None)) if
        i in [10, 17, 24, 31]),
    "ST-LRP-C-s54321": lambda model: lambda x, y: multi_interpolate(
        hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer_num=None)) if
        i in [5, 10, 17, 24, 31]),
    # "SIG-LRP-0-s54": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig', method='lrp0', layer_num=None)) if
    #     i in [24, 31]),
    # "SIG-LRP-0-s543": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig', method='lrp0', layer_num=None)) if
    #     i in [17, 24, 31]),
    # "SIG-LRP-0-s5432": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig', method='lrp0', layer_num=None)) if
    #     i in [10, 17, 24, 31]),
    # "SIG-LRP-0-s54321": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig', method='lrp0', layer_num=None)) if
    #     i in [5, 10, 17, 24, 31]),
    # "SIG-LRP-ZP-s54": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig', method='lrpzp', layer_num=None)) if
    #     i in [24, 31]),
    # "SIG-LRP-ZP-s543": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig', method='lrpzp', layer_num=None)) if
    #     i in [17, 24, 31]),
    # "SIG-LRP-ZP-s5432": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig', method='lrpzp', layer_num=None)) if
    #     i in [10, 17, 24, 31]),
    # "SIG-LRP-ZP-s54321": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig', method='lrpzp', layer_num=None)) if
    #     i in [5, 10, 17, 24, 31]),
    "SIG-LRP-C-s54": lambda model: lambda x, y: multi_interpolate(
        hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig', method='lrpc', layer_num=None)) if
        i in [24, 31]),
    "SIG-LRP-C-s543": lambda model: lambda x, y: multi_interpolate(
        hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig', method='lrpc', layer_num=None)) if
        i in [17, 24, 31]),
    "SIG-LRP-C-s5432": lambda model: lambda x, y: multi_interpolate(
        hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig', method='lrpc', layer_num=None)) if
        i in [10, 17, 24, 31]),
    "SIG-LRP-C-s54321": lambda model: lambda x, y: multi_interpolate(
        hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig', method='lrpc', layer_num=None)) if
        i in [5, 10, 17, 24, 31]),

    "SIG-LID-Taylor-s54": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5, 4), lin=True, bp='sig'),
    "SIG-LID-Taylor-s543": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5, 4, 3), lin=True, bp='sig'),
    "SIG-LID-Taylor-s5432": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5, 4, 3, 2), lin=True,bp='sig'),
    "SIG-LID-Taylor-s54321": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5, 4, 3, 2, 1), lin=True,bp='sig'),
    "SIG-LID-IG-s54": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5, 4), lin=False,bp='sig'),
    "SIG-LID-IG-s543": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5, 4, 3), lin=False,bp='sig'),
    "SIG-LID-IG-s5432": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5, 4, 3, 2), lin=False,bp='sig'),
    "SIG-LID-IG-s54321": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5, 4, 3, 2, 1), lin=False,bp='sig'),

    # "SIG-LID-Taylor-s54-r0": lambda model: lambda x, y: LID_m_caller(model, x, y, x0='zero', s=(5, 4), lin=True, bp='sig'),
    # "SIG-LID-Taylor-s543-r0": lambda model: lambda x, y: LID_m_caller(model, x, y, x0='zero', s=(5, 4, 3), lin=True, bp='sig'),
    # "SIG-LID-Taylor-s5432-r0": lambda model: lambda x, y: LID_m_caller(model, x, y, x0='zero', s=(5, 4, 3, 2), lin=True, bp='sig'),
    # "SIG-LID-Taylor-s54321-r0": lambda model: lambda x, y: LID_m_caller(model, x, y, x0='zero', s=(5, 4, 3, 2, 1), lin=True, bp='sig'),
    # "SIG-LID-IG-s54-r0": lambda model: lambda x, y: LID_m_caller(model, x, y, x0='zero', s=(5, 4), lin=False, bp='sig'),
    # "SIG-LID-IG-s543-r0": lambda model: lambda x, y: LID_m_caller(model, x, y, x0='zero', s=(5, 4, 3), lin=False, bp='sig'),
    # "SIG-LID-IG-s5432-r0": lambda model: lambda x, y: LID_m_caller(model, x, y, x0='zero', s=(5, 4, 3, 2), lin=False, bp='sig'),
    # "SIG-LID-IG-s54321-r0": lambda model: lambda x, y: LID_m_caller(model, x, y, x0='zero', s=(5, 4, 3, 2, 1), lin=False, bp='sig'),

    # "SIG-LID-Taylor-CE05-s54": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5, 4), bp='sig', lin=1, ce=0.5),
    # "SIG-LID-Taylor-CE05-s543": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5, 4, 3),bp='sig',lin=1,ce=0.5),
    # "SIG-LID-Taylor-CE05-s5432": lambda model: lambda x, y: LID_m_caller(model, x, y,s=(5,4,3,2),bp='sig',lin=1,ce=0.5),
    # "SIG-LID-Taylor-CE05-s54321": lambda model: lambda x, y:LID_m_caller(model,x,y,s=(5,4,3,2,1),bp='sig',lin=1,ce=0.5),
    # "SIG-LID-IG-CE02-s54": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5, 4), bp='sig', lin=0, ce=0.2),
    # "SIG-LID-IG-CE02-s543": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5, 4, 3), bp='sig', lin=0, ce=0.2),
    # "SIG-LID-IG-CE02-s5432": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5,4,3,2), bp='sig', lin=0, ce=0.2),
    # "SIG-LID-IG-CE02-s54321": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5,4,3,2,1),bp='sig',lin=0,ce=0.2),
    # "SIG-LID-IG-CE05-s54": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5, 4), bp='sig', lin=0, ce=0.5),
    # "SIG-LID-IG-CE05-s543": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5, 4, 3), bp='sig', lin=0, ce=0.5),
    # "SIG-LID-IG-CE05-s5432": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5,4,3,2), bp='sig', lin=0, ce=0.5),
    # "SIG-LID-IG-CE05-s54321": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5,4,3,2,1),bp='sig',lin=0,ce=0.5),
    # "SIG-LID-IG-CE10-s54": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5, 4), bp='sig', lin=0, ce=1.0),
    # "SIG-LID-IG-CE10-s543": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5, 4, 3), bp='sig', lin=0, ce=1.0),
    # "SIG-LID-IG-CE10-s5432": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5,4,3,2), bp='sig', lin=0, ce=1.0),
    # "SIG-LID-IG-CE10-s54321": lambda model: lambda x, y: LID_m_caller(model, x, y, s=(5,4,3,2,1),bp='sig',lin=0,ce=1.0),
}
