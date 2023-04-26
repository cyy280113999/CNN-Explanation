import torch.utils.data as TD
from datasets.bbox_imgnt import BBImgnt
from datasets.rrcri import RRCRI
from datasets.ri import RI
from datasets.DiscrimDataset import DiscrimDataset
from Evaluators.ProbChangeEvaluator import ProbChangeEvaluator
from Evaluators.MaximalPatchEvaluator import MaximalPatchEvaluator
from Evaluators.PointGameEvaluator import PointGameEvaluator
from Evaluators.ProbChangePlus import ProbChangePlusEvaluator
from HeatmapMethods import *


def rand_choice(ds):
    import time
    np.random.seed(1)
    torch.random.manual_seed(1)
    num_samples = 5000
    dslen=len(ds)
    indices = np.random.choice(dslen, num_samples)
    ds = TD.Subset(ds, indices)
    np.random.seed(int(time.time()))
    torch.random.manual_seed(int(time.time()))
    return ds


dataset_callers = {  # creating when called
    # ==imgnt val
    'sub_imgnt': lambda: rand_choice(getImageNet('val')),
    # ==discrim ds
    'DiscrimDataset': lambda: DiscrimDataset(),
    # =relabeled imgnt
    'relabeled_top0': lambda: rand_choice(RI(topk=0)),
    'relabeled_top1': lambda: rand_choice(RI(topk=1)),
    # ==bbox imgnt
    'bbox_imgnt': lambda: rand_choice(BBImgnt()),
}
models = {
    'vgg16': lambda: get_model('vgg16'),
    'resnet34': lambda: get_model('resnet34'),
}
# settings
ds_name = 'sub_imgnt'
model_name = 'vgg16'
EvalClass = ProbChangePlusEvaluator

eval_vis_check = False
eval_heatmap_methods = {
    # base-line : cam, lrp top layer
    # "GradCAM-f": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, -1)).__call__,
    #                                    sg=False, relu=True),
    # "GradCAM-origin-f": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, '-1')).__call__,
    #                                           sg=False, relu=False),
    # "SG-GradCAM-origin-f": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, '-1')).__call__,
    #                                              sg=True, relu=False),
    # "LayerCAM-f": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '-1')).__call__,
    #                                            sg=False, relu_weight=True, relu=True),
    # "ScoreCAM-f": lambda model: lambda x, y: ScoreCAM(model, '-1')(x, y, sg=True, relu=False),
    # "LRP-0-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpc', layer='-1')),
    # "SG-LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer=-1)),
    "IG-f": lambda model: lambda x, y: interpolate_to_imgsize(
        IGDecomposer(model, x, y, layer_name=('features', -1), post_softmax=False)),
    "SIG-f": lambda model: lambda x, y: interpolate_to_imgsize(
        IGDecomposer(model, x, y, layer_name=('features', -1), post_softmax=True)),

    # base-line : unimportant part
    # "Random": lambda model: lambda x,y: normalize_R(torch.randn((1,)+x.shape[-2:])),
    # "AblationCAM-f": lambda model: lambda x, y: AblationCAM(model, -1)(x, y, relu=False),
    # "RelevanceCAM-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     RelevanceCAM(model)(x, y, backward_init='c', method='lrpzp', layer=-1)),
    # "LRP-ZP-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpzp', layer=-1)),
    # "SG-LRP-ZP-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp', layer=-1)),

    # improvement
    # "ST-LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer=-1)),
    # "SIG0-LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=-1)),
    #
    # "ST-LRP-ZP-31": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpzp', layer=31).sum(1, True)),
    # "SIG0-LRP-ZP-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpzp', layer=-1)),

    # Increment Decomposition
    # "LID-Taylor-5": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=5, bp=None, linear=True),
    # "LID-Taylor-sig-5": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=5, bp='sig', linear=True),
    # "LID-IG-5": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=5, bp=None, linear=False),
    # "LID-IG-sig-5": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=5, bp='sig', linear=False),

    # mix
    # "LID-Taylor-sig-54": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5, 4), linear=True,
    #                                                              bp='sig'),
    # "LID-Taylor-sig-543": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5, 4, 3), linear=True,
    #                                                               bp='sig'),
    # "LID-Taylor-sig-5432": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5, 4, 3, 2), linear=True,
    #                                                                bp='sig'),
    # "LID-IG-sig-54": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5, 4), linear=False,
    #                                                          bp='sig'),
    # "LID-IG-sig-543": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5, 4, 3), linear=False,
    #                                                           bp='sig'),
    # "LID-IG-sig-5432": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5, 4, 3, 2), linear=False,
    #                                                            bp='sig'),
    # "LID-IG-sig-54321": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5, 4, 3, 2, 1), linear=False,
    #                                                             bp='sig'),

    # base-line : pixel layer
    # "SG-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer=1)),
    # "SG-LRP-ZP-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp', layer=1)),
    # "IG": lambda model: lambda x, y: interpolate_to_imgsize(
    #     IGDecomposer(model)(x, y)),

    # pixel level
    # "ST-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer=1)),
    # "SIG0-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=1)),
    # "LID-Taylor-sig-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LIDLinearDecomposer(model)(x, y, layer=1, backward_init='sig')),
    # "LID-IG-sig-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LIDIGDecomposer(model)(x, y, layer=1, backward_init='sig')),

    # mix
    # "SIG0-LRP-C-m": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=None))
    #     if i in [1, 5, 10, 17, 24]),
    # "LID-IG-sig-m": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LIDIGDecomposer(model)(x, y, layer=None, backward_init='sig'))
    #     if i in [1, 5, 10, 17, 24]),
    # "LID-Taylor-sig-m": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LIDLinearDecomposer(model)(x, y, layer=None, backward_init='sig'))
    #     if i in [24, 31]),

    # differ layer
    # "LID-IG-sig-24": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LIDIGDecomposer(model)(x, y, layer=24, backward_init='sig')),
    # "LID-IG-sig-17": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LIDIGDecomposer(model)(x, y, layer=17, backward_init='sig')),
    # "LID-IG-sig-10": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LIDIGDecomposer(model)(x, y, layer=10, backward_init='sig')),
    # "LID-IG-sig-5": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LIDIGDecomposer(model)(x, y, layer=5, backward_init='sig')),
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
