from functools import partial
import torch
import torch.nn.functional as nf
from utils import *
from methods.cam.gradcam import GradCAM
from methods.cam.layercam import LayerCAM
from methods.scorecam import ScoreCAM
from methods.AblationCAM import AblationCAM
from methods.RelevanceCAM import RelevanceCAM
from methods.Taylor_0 import Taylor_0
from methods.LRP_0 import LRP_0
from methods.LRP import LRP_Generator
from methods.IG import IGDecomposer
from methods.LIDLinearDecompose import LIDLinearDecomposer
from methods.LIDIGDecompose import LIDIGDecomposer
from methods.LIDDecomposer_beta import LIDDecomposer

# partial fun 参数是静态的，传了就不能变，此处要求每次访问self.model。（写下语句的时候就创建完了）
# lambda fun 是动态的，运行时解析
# 结合一下匿名lambda函数就可以实现 创建含动态参数(model)的partial fun，只多了一步调用()
cam_model_dict_by_layer = lambda model, layer: {'arch': model, 'layer_name': f'{layer}'}
interpolate_to_imgsize = lambda x: heatmapNormalizeR(nf.interpolate(x.sum(1, True), 224, mode='bilinear'))
multi_interpolate = lambda xs: heatmapNormalizeR(
    sum(heatmapNormalizeR(nf.interpolate(x.sum(1, True), 224, mode='bilinear')) for x in xs))


def RelevanceExtractor(model,layer_names=(None,)):
    layer=model
    for l in layer_names:
        if isinstance(l,int):# for sequential
            layer=layer[l]
        elif isinstance(l,str) and hasattr(layer,l):# for named child
            layer=layer.l
        else:
            raise Exception()
    return layer.Ry

def LID_VGG_m_caller_beta(model, x, y, which_=(23, 30), linear=False, bp='sig'):
    d = LIDDecomposer(model,LINEAR=linear, DEFAULT_STEP=11)
    d.forward(x)
    r = d.backward(y, bp)
    hm = multi_interpolate(RelevanceExtractor('features', i) for i in which_)
    return hm

def LID_VGG_m_caller(model, x, y, which_=(23, 30), linear=False, bp='sig'):
    d = LIDDecomposer(model,LINEAR=linear, DEFAULT_STEP=11)
    d.forward(x)
    r = d.backward(y, bp)
    hm = multi_interpolate([model.features[i].Ry for i in which_])
    return hm


def LID_Res34_m_caller(model, x, y, which_=(0, 1, 2, 3, 4), linear=False, bp='sig'):
    d = LIDDecomposer(model,LINEAR=linear)
    d.forward(x)
    r = d.backward(y, bp)
    hms = [model.maxpool.Ry, model.layer1[-1].relu2.Ry,
           model.layer2[-1].relu2.Ry, model.layer3[-1].relu2.Ry,
           model.layer4[-1].relu2.Ry]
    hm = multi_interpolate([hms[i] for i in which_])
    return hm


def LID_Res50_m_caller(model, x, y, which_=(0, 1, 2, 3, 4), linear=False, bp='sig'):
    d = LIDDecomposer(model,LINEAR=linear)
    d.forward(x)
    r = d.backward(y, bp)
    hms = [model.maxpool.Ry, model.layer1[-1].relu3.Ry,
           model.layer2[-1].relu3.Ry, model.layer3[-1].relu3.Ry,
           model.layer4[-1].relu3.Ry]
    hm = multi_interpolate([hms[i] for i in which_])
    return hm


# the method interface, all methods must follow this:
# the method can call twice
# 1. the method first accept "model" parameter, create a callable function "_m = m(model)"
# 2. the heatmap generated by secondly calling "hm = _m(x,yc)"
heatmap_methods = {
    "None": None,

    "Res34-LID-IG-sig-1234": lambda model: lambda x, y: LID_Res34_m_caller(model, x, y, (1, 2, 3, 4), linear=False, bp='sig'),
    "Res34-LID-IG-sig-234": lambda model: lambda x, y: LID_Res34_m_caller(model, x, y, (2, 3, 4), linear=False, bp='sig'),
    "Res34-LID-IG-sig-34": lambda model: lambda x, y: LID_Res34_m_caller(model, x, y, (3, 4), linear=False, bp='sig'),
    "Res34-LID-IG-sig-4": lambda model: lambda x, y: LID_Res34_m_caller(model, x, y, (4,), linear=False, bp='sig'),


    # "LID-Res50-sig-1234": lambda model: lambda x, y: LID_Res50_m_caller(model, x, y, (1, 2, 3, 4)),
    # "LID-Res50-sig-234": lambda model: lambda x, y: LID_Res50_m_caller(model, x, y, (2, 3, 4)),
    # "LID-Res50-sig-34": lambda model: lambda x, y: LID_Res50_m_caller(model, x, y, (3, 4)),

    # -----------CAM
    # --cam method using layer: 8,9,15,16,22,23,29,30
    # "GradCAM-f": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, '-1')).__call__, sg=False,
    #                                    relu=True),  # cam can not auto release, so use partial
    # "GradCAM-origin-f": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, '-1')).__call__, sg=False,
    #                                           relu=False),
    # "SG-GradCAM-origin-f": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, '-1')).__call__,
    #                                              sg=True, relu=False),
    # "GradCAM-origin-29": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, '29')).__call__,sg=False, relu=False),
    # "SG-GradCAM-origin-29": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, '29')).__call__,sg=True, relu=False),
    # GradCAM 23 is nonsense

    # --LayerCAM-origin-f == LRP-0-f
    # "LayerCAM-origin-f": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '-1')).__call__,
    #                                            sg=False, relu_weight=False, relu=False),
    # "LRP-0-f-grad": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_0(model, x, y, Relevance_Propagate=False)[31]),
    # "LRP-0-f-relev": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_0(model, x, y, Relevance_Propagate=True)[31]),
    # "LayerCAM-f": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '-1')).__call__,
    #                                     sg=False, relu_weight=True, relu=True),
    # "LayerCAM-origin-29": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '29')).__call__,sg=False, relu=False),
    # "LayerCAM-origin-23": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '23')).__call__,sg=False, relu=False),
    # "LayerCAM-origin-22": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '22')).__call__,sg=False, relu=False),
    # --SG LayerCAM-origin-f == ST-LRP-0-f
    # "SG-LayerCAM-origin-f": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '-1')).__call__,sg=True, relu=False),

    # "SG-LayerCAM-origin-29": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '29')).__call__,
    #                                   sg=True, relu=False),
    # "SG-LayerCAM-origin-23": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '23')).__call__,
    #                                   sg=True, relu=False),
    # "SG-LayerCAM-origin-22": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '22')).__call__,
    #                                   sg=True, relu=False),
    # "SG-LayerCAM-origin-16": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '16')).__call__,
    #                                                sg=True, relu=False),
    # "SG-LayerCAM-origin-0": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '0')).__call__,
    #                                   sg=True, relu=False),

    # --others
    "Random": lambda model: lambda x, y: heatmapNormalizeR(torch.randn((1,) + x.shape[-2:])),
    "ScoreCAM-f": lambda model: lambda x, y: ScoreCAM(model, '-1')(x, y, sg=True, relu=False),
    "AblationCAM-f": lambda model: lambda x, y: AblationCAM(model, -1)(x, y, relu=False),
    "RelevanceCAM-f": lambda model: lambda x, y: interpolate_to_imgsize(
        RelevanceCAM(model)(x, y, backward_init='c', method='lrpzp', layer=-1)),
    # "RelevanceCAM-24": lambda model: lambda x, y: interpolate_to_imgsize(
    #     RelevanceCAM(model)(x, y, backward_init='c', method='lrpzp', layer=24)),
    # "RelevanceCAM-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     RelevanceCAM(model)(x, y, backward_init='c', method='lrpzp', layer=1)),
    # "Taylor-30": lambda model:lambda x, y: interpolate_to_imgsize(Taylor(model, 30)(x, y)),

    # ------------LRP Top
    # # LRP-C use LRP-0 in classifier
    "LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='normal', method='lrpc', layer=-1)),
    # # LRP-Z is nonsense
    # "LRP-Z-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpz', layer=-1)
    #     .sum(1, True)),# lrpz 31 is bad
    # # LRP-ZP no edge highlight
    "LRP-ZP-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='normal', method='lrpzp', layer=-1)),
    # # LRP-W2 all red
    # "LRP-W2-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpw2', layer=-1)
    #     .sum(1, True)),
    "SIG0-LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=-1)),
    "SIGP-LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='sigp', method='lrpc', layer=-1)),
    "SG-LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer=-1)),
    "ST-LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer=-1)),

    # "SIG0-LRP-ZP-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpzp', layer=-1)),
    # "SIGP-LRP-ZP-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sigp', method='lrpzp', layer=-1)),
    # "SG-LRP-ZP-31": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp', layer=31).sum(1, True)),
    # "ST-LRP-ZP-31": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpzp', layer=31).sum(1, True)),
    # # to bad often loss discrimination
    # "C-LRP-C 31": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='c', method='lrpc', layer=31).sum(1, True)),
    # "C-LRP-ZP 31": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='c', method='lrpzp', layer=31).sum(1, True)),

    # ---------LRP-middle
    # "LRP C 30": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpc',layer=30).sum(1, True)),
    # "LRP C 24": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpc', layer=24).sum(1, True)),
    # "LRP C 23": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpc')
    #     [23].sum(1, True), 224, mode='bilinear'),
    # "SG LRP C 30": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpc',layer=30).sum(1, True),
    #     224, mode='bilinear'),
    # "SG LRP C 24": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpc')
    #     [24].sum(1, True), 224, mode='bilinear'),
    # "SG LRP C 23": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpc')
    #     [23].sum(1, True), 224, mode='bilinear'),
    # "ST LRP C 24": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpc')
    #     [24].sum(1, True), 224, mode='bilinear'),
    # "ST LRP C 23": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpc')
    #     [23].sum(1, True), 224, mode='bilinear'),

    "SIG0-LRP-C-24": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=24)),
    "SIG0-LRP-C-17": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=17)),
    "SIG0-LRP-C-10": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=10)),
    "SIG0-LRP-C-5": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=5)),

    # "LRP ZP 31": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpzp')
    #     [31].sum(1, True), 224, mode='bilinear'),
    # "LRP ZP 30": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpzp')
    #     [30].sum(1, True), 224, mode='bilinear'),
    # "LRP ZP 24": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpzp')
    #     [24].sum(1, True), 224, mode='bilinear'),
    # "LRP ZP 23": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpzp')
    #     [23].sum(1, True), 224, mode='bilinear'),
    # "SG LRP ZP 31": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp')
    #     [31].sum(1, True), 224, mode='bilinear'),
    # "SG LRP ZP 30": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp')
    #     [30].sum(1, True), 224, mode='bilinear'),
    # "SG LRP ZP 24": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp')
    #     [24].sum(1, True), 224, mode='bilinear'),
    # "SG LRP ZP 23": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp')
    #     [23].sum(1, True), 224, mode='bilinear'),
    # "ST LRP ZP 31": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpzp')
    #     [31].sum(1, True), 224, mode='bilinear'),
    # "ST LRP ZP 30": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpzp')
    #     [30].sum(1, True), 224, mode='bilinear'),
    # "ST LRP ZP 24": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpzp')
    #     [24].sum(1, True), 224, mode='bilinear'),
    # "ST LRP ZP 23": lambda model: lambda x, y: nf.interpolate(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpzp')
    #     [23].sum(1, True), 224, mode='bilinear'),

    # ----------LRP-pixel
    "LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='normal', method='lrpc', layer=1)),
    "LRP-C-0": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='normal', method='lrpc', layer=0)),
    "SG-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer=1)),
    "SG-LRP-C-0": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer=0)),
    "ST-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer=1)),
    "ST-LRP-C-0": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer=0)),
    "SIG0-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=1)),
    "SIG0-LRP-C-0": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=0)),
    # # noisy
    # "LRP-0 0": lambda model: lambda x, y: LRP_Generator(model)(
    #     x, y, backward_init='normal', method='lrp0', layer=0).sum(1, True),
    # # nonsense
    # "LRP-Z 0": lambda model: lambda x, y: LRP_Generator(model)(
    #     x, y, backward_init='normal', method='lrpz', layer=0).sum(1, True),
    # # noisy
    # "S-LRP-C 1": lambda model: lambda x, y: LRP_Generator(model)(
    #     x, y, backward_init='normal', method='slrp', layer=1).sum(1, True),
    # "S-LRP-C 0": lambda model: lambda x, y: LRP_Generator(model)(
    #     x, y, backward_init='normal', method='slrp', layer=0).sum(1, True),
    # "LRP-ZP 0": lambda model: lambda x, y: LRP_Generator(model)(
    #     x, y, backward_init='normal', method='lrpzp', layer=0).sum(1, True),
    "SG-LRP-ZP-0": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp', layer=0)),
    "ST-LRP-ZP-0": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='st', method='lrpzp', layer=0)),

    # IG
    "IG": lambda model: lambda x, y: interpolate_to_imgsize(
        IGDecomposer(model)(x, y, post_softmax=False)),
    "SIG": lambda model: lambda x, y: interpolate_to_imgsize(
        IGDecomposer(model)(x, y, post_softmax=True)),

    # -----------Increment Decomposition
    # LID-linear?-init-middle-end.
    # LID-Taylor-sig-f means it is layer linear decompose, given sig init , ending at feature layer
    # LID-IG-sig-1 means it is layer integrated decompose, given sig init , ending at layer-1
    "LID-Taylor-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LIDLinearDecomposer(model)(x, y, layer=-1)),
    # "LID-Taylor-f-relev": lambda model: lambda x, y: interpolate_to_imgsize(# equivalent
    #     LIDLinearDecomposer(model)(x, y, layer=-1, Relevance_Propagate=True)),
    "LID-Taylor-sig-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LIDLinearDecomposer(model)(x, y, layer=-1, backward_init='sig')),

    # "LID-IG-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LIDIGDecomposer(model)(x, y, layer=-1, backward_init='normal')),
    "LID-IG-sig-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LIDIGDecomposer(model)(x, y, layer=-1, backward_init='sig')),

    # "LID-Taylor-1": lambda model: lambda x, y: interpolate_to_imgsize(#noisy
    #     LIDLinearDecomposer(model)(x, y, layer=1)),
    # "LID-Taylor-sig-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LIDLinearDecomposer(model)(x, y, layer=1, backward_init='sig')),

    # "LID-IG-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LIDIGDecomposer(model)(x, y, layer=1, backward_init='normal')),
    "LID-IG-sig-1": lambda model: lambda x, y: interpolate_to_imgsize(
        LIDIGDecomposer(model)(x, y, layer=1, backward_init='sig')),
    # "LID-sig-1-beta": lambda model: lambda x, y: LID_VGG_m_caller(model, x, y,which_=(0,),linear=False,bp='sig'),

    "LID-IG-sig-24": lambda model: lambda x, y: interpolate_to_imgsize(
        LIDIGDecomposer(model)(x, y, layer=24, backward_init='sig')),
    "LID-IG-sig-17": lambda model: lambda x, y: interpolate_to_imgsize(
        LIDIGDecomposer(model)(x, y, layer=17, backward_init='sig')),
    "LID-IG-sig-10": lambda model: lambda x, y: interpolate_to_imgsize(
        LIDIGDecomposer(model)(x, y, layer=10, backward_init='sig')),
    "LID-IG-sig-5": lambda model: lambda x, y: interpolate_to_imgsize(
        LIDIGDecomposer(model)(x, y, layer=5, backward_init='sig')),

    # mix methods
    "SIG0-LRP-C-m": lambda model: lambda x, y: multi_interpolate(
        hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=None))
        if i in [1, 5, 10, 17, 24]),
    # "LID-Taylor-sig-m": lambda model: lambda x, y: multi_interpolate(#noisy
    #     hm for i, hm in enumerate(LIDLinearDecomposer(model)(x, y, layer=None, backward_init='sig'))
    #     if i in [24, 31]),
    "LID-IG-sig-m": lambda model: lambda x, y: multi_interpolate(
        hm for i, hm in enumerate(LIDIGDecomposer(model)(x, y, layer=None, backward_init='sig'))
        if i in [1, 5, 10, 17, 24]),

}
