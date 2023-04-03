from HeatmapMethods import *
from Evaluators.ProbChangeEvaluator import ProbChangeEvaluator
from Evaluators.MaximalPatchEvaluator import MaximalPatchEvaluator
from Evaluators.PointGameEvaluator import PointGameEvaluator

# settings
ds_name = 'sub_imgnt'
model_name = 'resnet34'
EvalClass = ProbChangeEvaluator

eval_heatmap_methods = {
    # resnet
    "ScoreCAM": lambda model: lambda x, y: ScoreCAM(model, 'layer4_-1')(x, y, sg=True, relu=False),

    "Res34-LID-T-sig-1234": lambda model: lambda x, y: LID_Res34_m_caller(model, x, y, (1, 2, 3, 4), linear=True, bp='sig'),
    "Res34-LID-T-sig-234": lambda model: lambda x, y: LID_Res34_m_caller(model, x, y, (2, 3, 4), linear=True, bp='sig'),
    "Res34-LID-T-sig-34": lambda model: lambda x, y: LID_Res34_m_caller(model, x, y, (3, 4), linear=True, bp='sig'),
    "Res34-LID-T-sig-4": lambda model: lambda x, y: LID_Res34_m_caller(model, x, y, (4,), linear=True, bp='sig'),

    "Res34-LID-IG-sg-1234": lambda model: lambda x, y: LID_Res34_m_caller(model, x, y, (1, 2, 3, 4), linear=False, bp='sg'),
    "Res34-LID-IG-sg-234": lambda model: lambda x, y: LID_Res34_m_caller(model, x, y, (2, 3, 4), linear=False, bp='sg'),
    "Res34-LID-IG-sg-34": lambda model: lambda x, y: LID_Res34_m_caller(model, x, y, (3, 4), linear=False, bp='sg'),
    "Res34-LID-IG-sg-4": lambda model: lambda x, y: LID_Res34_m_caller(model, x, y, (4,), linear=False, bp='sg'),

    # "Res34-LID-IG-sig-1234": lambda model: lambda x, y: LID_Res34_m_caller(model, x, y, (1, 2, 3, 4), linear=False, bp='sig'),
    # "Res34-LID-IG-sig-234": lambda model: lambda x, y: LID_Res34_m_caller(model, x, y, (2, 3, 4), linear=False, bp='sig'),
    # "Res34-LID-IG-sig-34": lambda model: lambda x, y: LID_Res34_m_caller(model, x, y, (3, 4), linear=False, bp='sig'),
    # "Res34-LID-IG-sig-4": lambda model: lambda x, y: LID_Res34_m_caller(model, x, y, (4,), linear=False, bp='sig'),


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

    # base-line : unimportant part
    # "Random": lambda model: lambda x,y: normalize_R(torch.randn((1,)+x.shape[-2:])),
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