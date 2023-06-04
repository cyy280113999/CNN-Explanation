from functools import partial
import torch
import torch.nn.functional as nf

from methods.LRPG import LRPWithGradient
from utils import *
from methods.cam.gradcam import GradCAM
from methods.cam.layercam import LayerCAM
from methods.scorecam import ScoreCAM
from methods.AblationCAM import AblationCAM
from methods.RelevanceCAM import RelevanceCAM
from methods.LRP import LRP_Generator
from methods.IG import IGDecomposer
from methods.LIDDecomposer import *


# partial fun 参数是静态的，传了就不能变，此处要求每次访问self.model。（写下语句的时候就创建完了）
# lambda fun 是动态的，运行时解析
# 结合一下匿名lambda函数就可以实现 创建含动态参数(model)的partial fun，只多了一步调用()
def method_caller(method_class, *args, **kwargs):
    def method(model):
        method = method_class(model, **kwargs)
        return partial(method.__call__, **kwargs)
    return method
cam_dict = lambda model, layer: {'arch': model, 'layer_name': layer}


# the method interface, all methods must follow this:
# the method can call twice
# 1. the method first accept "model" parameter, create a callable function "_m = m(model)"
# 2. the heatmap generated by secondly calling "hm = _m(x,yc)"
heatmap_methods = {
    "None": None,
    # ========= temp test

    # ============================== Top level features
    # ---------- CAM series
    # -- vgg features stages: 0,4,9,16,23,30
    # "GradCAM-f": lambda model: partial(GradCAM(cam_dict(model, None)).__call__, sg=False,
    #                                    relu=True),  # cam can not auto release, so use partial
    "GradCAM-origin-f": lambda model: partial(GradCAM(cam_dict(model, None)).__call__, sg=False, relu=False),
    # "SG-GradCAM-origin-f": lambda model: partial(GradCAM(cam_dict(model, None)).__call__,
    #                                              sg=True, relu=False),
    # "LayerCAM-f": lambda model: partial(LayerCAM(cam_dict(model, None)).__call__,
    #                                     sg=False, relu_weight=True, relu=True),
    # --LayerCAM-origin-f == LRP-0-f
    # "LayerCAM-origin-f": lambda model: partial(LayerCAM(cam_dict(model, None)).__call__,
    #                                            sg=False, relu_weight=False, relu=False),
    # --SG LayerCAM-origin-f == ST-LRP-0-f
    # "SG-LayerCAM-origin-f": lambda model: partial(LayerCAM(cam_dict(model, None)).__call__,sg=True, relu=False),
    "ScoreCAM-f": lambda model: partial(ScoreCAM(model, None).__call__, sg=True, relu=False),
    # ------------- LRP series
    # -- LRP-C use LRP-0 in classifier, they are equivalent.
    "LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='normal', method='lrpc', layer_num=-1)),
    # # LRP-Z is nonsense
    # "LRP-Z-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpz', layer=-1)
    #     .sum(1, True)),# lrpz 31 is bad
    # # LRP-ZP no edge highlight
    "LRP-ZP-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='normal', method='lrpzp', layer_num=-1)),
    # # LRP-W2 all red
    # "LRP-W2-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpw2', layer=-1)
    #     .sum(1, True)),
    # ---- contrastive LRP
    # # to bad often loss discrimination
    # "C-LRP-C 31": lambda model: lambda x, y: interpolate_to_imgsize(  # c is unstable
    #     LRP_Generator(model)(x, y, backward_init='c', method='lrpc', layer_num=31).sum(1, True)),
    "C-LRP-C2 31": lambda model: lambda x, y: interpolate_to_imgsize(  # c2 is stable, interesting.
        LRP_Generator(model)(x, y, backward_init='c', method='lrpc2', layer_num=31).sum(1, True)),
    # "C-LRP-C 0": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='c', method='lrpc', layer_num=0).sum(1, True)),
    # "C-LRP-C2 0": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='c', method='lrpc2', layer_num=0).sum(1, True)),
    # "C-LRP-ZP 31": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='c', method='lrpzp', layer=31).sum(1, True)),
    "SG-LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer_num=-1)),
    # "SG-LRP-ZP-31": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp', layer=31).sum(1, True)),

    # ------------------ our new Layer-wise Increment Decomposition (LID) Framework
    # ------ our improvement: increment(in middle/contrast), contrast, nonlinear(in middle/contrast)
    # ----- increment(in contrast)
    "ST-LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer_num=-1)),
    # ----- nonlinear in contrast
    "SIG-LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
        LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer_num=-1)),
    # ----- full nonlinear with contrast but not increment in middle
    # "LRP-IG-f": lambda model: lambda x, y: interpolate_to_imgsize(  # unstable all layers due to low values
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpig', layer=-1)),
    # "SIG-LRP-IG-f": lambda model: lambda x, y: interpolate_to_imgsize(  # unstable
    #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpig', layer=-1)),

    # "ST-LRP-ZP-31": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpzp', layer=31).sum(1, True)),
    # "SIG-LRP-ZP-f": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpzp', layer=-1)),

    # LID series name convention: Init-LID-Type-Layer.
    # SIG-LID-Taylor-f means it is linear decompose, given sig init, ending at top feature layer
    # using 5 stage to refer different model layers:
    # VGG: 12345-> 5,10,17,24,31
    # Res: 12345-> mp,l1,l2,l3,l4
    # goog
    # ------- increment in middle
    "LID-Taylor-s5": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=5, bp=None, linear=True),
    # -------- nonlinear increment in middle
    "LID-IG-s5": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=5, bp=None, linear=False),
    # "ST-LID-Taylor-s5": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=5, bp='st', linear=True),
    # "ST-LID-IG-s5": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=5, bp='st', linear=False),
    # -------  full increment with nonlinear contrast
    "SIG-LID-Taylor-s5": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=5, bp='sig', linear=True),
    # ------- contrast, all increment, all nonlinear
    "SIG-LID-IG-s5": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=5, bp='sig', linear=False),


    # ---- others
    # "Random": lambda model: lambda x, y: heatmapNormalizeR(torch.randn((1,) + x.shape[-2:])),
    "AblationCAM-f": lambda model: lambda x, y: AblationCAM(model, -1)(x, y, relu=False),
    "RelevanceCAM-f": lambda model: lambda x, y: interpolate_to_imgsize(
        RelevanceCAM(model)(x, y, backward_init='c', method='lrpzp', layer_num=-1)),
    # IG
    "IG-f": lambda model: lambda x, y: interpolate_to_imgsize(
        IGDecomposer(model, x, y, layer_name=('features', -1), post_softmax=False)),
    "SIG-f": lambda model: lambda x, y: interpolate_to_imgsize(
        IGDecomposer(model, x, y, layer_name=('features', -1), post_softmax=True)),

    # ===================== Middle level features
    # -- lrp layer == cam layer + 1
    # "GradCAM-origin-29": lambda model: partial(GradCAM(cam_dict(model, '29')).__call__,sg=False, relu=False),
    # "SG-GradCAM-origin-29": lambda model: partial(GradCAM(cam_dict(model, '29')).__call__,sg=True, relu=False),
    # GradCAM 23 is nonsense
    # "LayerCAM-origin-29": lambda model: partial(LayerCAM(cam_dict(model, '29')).__call__,sg=False, relu=False),
    # "LayerCAM-origin-23": lambda model: partial(LayerCAM(cam_dict(model, '23')).__call__,sg=False, relu=False),
    # "LayerCAM-origin-22": lambda model: partial(LayerCAM(cam_dict(model, '22')).__call__,sg=False, relu=False),
    # "SG-LayerCAM-origin-29": lambda model: partial(LayerCAM(cam_dict(model, '29')).__call__,
    #                                   sg=True, relu=False),
    # "SG-LayerCAM-origin-23": lambda model: partial(LayerCAM(cam_dict(model, '23')).__call__,
    #                                   sg=True, relu=False),
    # "SG-LayerCAM-origin-22": lambda model: partial(LayerCAM(cam_dict(model, '22')).__call__,
    #                                   sg=True, relu=False),
    # "SG-LayerCAM-origin-16": lambda model: partial(LayerCAM(cam_dict(model, '16')).__call__,
    #                                                sg=True, relu=False),
    # "SG-LayerCAM-origin-0": lambda model: partial(LayerCAM(cam_dict(model, '0')).__call__,
    #                                   sg=True, relu=False),

    # "SIG-LRP-C-24": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer_num=24)),
    # "SIG-LRP-C-17": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer_num=17)),
    # "SIG-LRP-C-10": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer_num=10)),
    # "SIG-LRP-C-5": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer_num=5)),

    # "ST-LID-Taylor-s4": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=4, bp='st', linear=True),
    # "ST-LID-IG-s4": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=4, bp='st', linear=False),
    # "ST-LID-Taylor-s3": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=3, bp='st', linear=True),
    # "ST-LID-IG-s3": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=3, bp='st', linear=False),
    # "ST-LID-Taylor-s2": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=2, bp='st', linear=True),
    # "ST-LID-IG-s2": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=2, bp='st', linear=False),

    "SIG-LID-Taylor-s4": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=4, bp='sig', linear=True),
    "SIG-LID-Taylor-s3": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=3, bp='sig', linear=True),
    "SIG-LID-Taylor-s2": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=2, bp='sig', linear=True),
    "SIG-LID-IG-s4": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=4, bp='sig', linear=False),
    "SIG-LID-IG-s3": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=3, bp='sig', linear=False),
    "SIG-LID-IG-s2": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=2, bp='sig', linear=False),


    # ============= pixel level heatmaps
    # "LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpc', layer_num=1)),
    # "LRP-C-0": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpc', layer_num=0)),
    # "SG-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer_num=1)),
    # "SG-LRP-C-0": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer_num=0)),
    # "ST-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer_num=1)),
    # "ST-LRP-C-0": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer_num=0)),
    # "SIG-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer_num=1)),
    # "SIG-LRP-C-0": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer_num=0)),
    # -- noisy
    # "LRP-0-0": lambda model: lambda x, y: LRP_Generator(model)(
    #     x, y, backward_init='normal', method='lrp0', layer=0).sum(1, True),
    # -- nonsense
    # "LRP-Z-0": lambda model: lambda x, y: LRP_Generator(model)(
    #     x, y, backward_init='normal', method='lrpz', layer=0).sum(1, True),
    # -- noisy
    # "S-LRP-C-1": lambda model: lambda x, y: LRP_Generator(model)(
    #     x, y, backward_init='normal', method='slrp', layer=1).sum(1, True),
    # "S-LRP-C-0": lambda model: lambda x, y: LRP_Generator(model)(
    #     x, y, backward_init='normal', method='slrp', layer=0).sum(1, True),
    # "LRP-ZP-0": lambda model: lambda x, y: LRP_Generator(model)(
    #     x, y, backward_init='normal', method='lrpzp', layer=0).sum(1, True),
    # "SG-LRP-ZP-0": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp', layer=0)),
    # "ST-LRP-ZP-0": lambda model: lambda x, y: interpolate_to_imgsize(
    #     LRP_Generator(model)(x, y, backward_init='st', method='lrpzp', layer=0)),

    # "IG-5": lambda model: lambda x, y: interpolate_to_imgsize(  # ig lower is noisy
    #     IGDecomposer(model, x, y, layer_name=('features', 4), post_softmax=False)),
    # "SIG-5": lambda model: lambda x, y: interpolate_to_imgsize(
    #     IGDecomposer(model, x, y, layer_name=('features', 4), post_softmax=True)),

    # -----------Increment Decomposition
    "LID-Taylor-s1": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=1, bp=None, linear=True),
    "SIG-LID-Taylor-s1": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=1, bp='sig', linear=True),
    "LID-IG-s1": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=1, bp=None, linear=False),
    "SIG-LID-IG-s1": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=1, bp='sig', linear=False),
    "SIG-LID-IG-s0": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=0, bp='sig', linear=False),

    # =============== mix scale features
    "SIG-LRP-C-s54321": lambda model: lambda x, y: multi_interpolate(
        hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig', method='lrpc', layer_num=None))
        if i in [1, 5, 10, 17, 24]),
    "SIG-LID-Taylor-s54": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5,4), linear=True,
                                                                 bp='sig'),
    "SIG-LID-Taylor-s543": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5,4,3), linear=True,
                                                                  bp='sig'),
    "SIG-LID-Taylor-s5432": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5,4,3,2), linear=True,
                                                                   bp='sig'),
    "SIG-LID-IG-s54": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5,4), linear=False,
                                                             bp='sig'),
    "SIG-LID-IG-s543": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5,4,3), linear=False,
                                                              bp='sig'),
    "SIG-LID-IG-s5432": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5,4,3,2), linear=False,
                                                               bp='sig'),
    "SIG-LID-IG-s54321": lambda model: lambda x, y: LID_m_caller(model, x, y, which_=(5,4,3,2,1), linear=False,
                                                                bp='sig'),

}
