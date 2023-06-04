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
    np.random.seed(1)  # this is repeatable
    torch.random.manual_seed(1)
    num_samples = 5000
    dslen = len(ds)
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
    'googlenet': lambda: get_model('googlenet'),
}
# settings
ds_name = 'sub_imgnt'
model_name = 'vgg16'
EvalClass = ProbChangePlusEvaluator

eval_vis_check = False
eval_heatmap_methods = {
    # base-line : cam, lrp top layer
    # "GradCAM-f": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, None)).__call__,
    #                                    sg=False, relu=True),
    # "GradCAM-origin-f": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, None)).__call__,
    #                                           sg=False, relu=False),
    # "SG-GradCAM-origin-f": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, None)).__call__,
    #                                              sg=True, relu=False),
    # "LayerCAM-f": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, None)).__call__,
    #                                            sg=False, relu_weight=True, relu=True),
    # "ScoreCAM-f": lambda model: partial(ScoreCAM(model, None).__call__, sg=True, relu=False),
    # "LRP-0-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpc', layer=-1)),
    # "SG-LRP-0-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer=-1)),
    # "IG-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     IGDecomposer(model, x, y, layer_name=('features', -1), post_softmax=False)),
    # "SIG-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     IGDecomposer(model, x, y, layer_name=('features', -1), post_softmax=True)),

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
    # "ST-LRP-0-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer=-1)),
    # "SIG-LRP-0-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig', method='lrpc', layer=-1)),
    #
    # "ST-LRP-ZP-31": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpzp', layer=31).sum(1, True)),
    # "SIG-LRP-ZP-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig', method='lrpzp', layer=-1)),

    # Increment Decomposition
    # "SIG-LID-Taylor-s5": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=5, bp='sig', linear=True),
    # "SIG-LID-IG-s5": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=5, bp='sig', linear=False),
    # "SIG-LID-IG-s4": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=4, bp='sig', linear=False),

    # mix
    # "SIG-LID-IG-s54": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5, 4), linear=False,
    #                                                          bp='sig'),
    # "SIG-LID-IG-s543": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5, 4, 3), linear=False,
    #                                                           bp='sig'),
    # "SIG-LID-IG-s5432": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5, 4, 3, 2), linear=False,
    #                                                            bp='sig'),
    # "SIG-LID-IG-s54321": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5, 4, 3, 2, 1), linear=False,
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
    # "SIG-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig', method='lrpc', layer=1)),

    # mix
    # "SIG-LRP-C-s54321": lambda model: lambda x, y: multi_interpolate(
    #     hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig', method='lrpc', layer=None))
    #     if i in [1, 5, 10, 17, 24]),

}
