import torchvision as tv
import timm
from torchvision.models import VGG, AlexNet, ResNet, GoogLeNet, VisionTransformer

device = 'cuda'


class model_names:
    vgg16 = 'vgg16'
    alexnet = 'alexnet'
    res18 = 'res18'
    res34 = 'res34'
    res50 = 'res50'
    res101 = 'res101'
    res152 = 'res152'
    googlenet = 'googlenet'
    inc1 = googlenet
    inc3 = 'inc3'
    inc4 = 'inc4'
    dens121 = 'des121'
    convnext = 'convnext'
    vit = 'vit'
    deit = 'deit'
    swin = 'swin'


def list_model_names__():
    l=set()
    for k,v in model_names.__dict__:
        l.add(v)
    return l


# return params of model
# list:[caller, kwargs]
# tv.models.vgg16(weights=tv.models.VGG16_Weights.DEFAULT)
available_models = {
    model_names.vgg16: [tv.models.vgg16, {'weights': tv.models.VGG16_Weights.DEFAULT}],
    model_names.alexnet: [tv.models.alexnet, {'weights': tv.models.AlexNet_Weights.DEFAULT}],
    model_names.res18: [tv.models.resnet18, {'weights': tv.models.ResNet18_Weights.DEFAULT}],
    model_names.res34: [tv.models.resnet34, {'weights': tv.models.ResNet34_Weights.DEFAULT}],
    model_names.res50: [tv.models.resnet50, {'weights': tv.models.ResNet50_Weights.DEFAULT}],
    model_names.res101: [tv.models.resnet101, {'weights': tv.models.ResNet101_Weights.DEFAULT}],
    model_names.res152: [tv.models.resnet152, {'weights': tv.models.ResNet152_Weights.DEFAULT}],
    model_names.googlenet: [tv.models.googlenet, {'weights': tv.models.GoogLeNet_Weights.DEFAULT}],
    model_names.dens121: [tv.models.densenet121, {'weights': tv.models.DenseNet121_Weights.DEFAULT}],
    model_names.convnext: [timm.models.create_model, {'model_name': 'convnext_tiny', 'pretrained': True}],
    model_names.inc3: [timm.models.create_model, {'model_name': 'inception_v3', 'pretrained': True}],
    model_names.inc4: [timm.models.create_model, {'model_name': 'inception_v4', 'pretrained': True}],
    model_names.vit: [timm.models.create_model, {'model_name': 'vit_base_patch16_224', 'pretrained': True}],
    model_names.deit: [timm.models.create_model, {'model_name': 'deit_base_patch16_224', 'pretrained': True}],
    model_names.swin: [timm.models.create_model, {'model_name': 'swin_base_patch4_window7_224', 'pretrained': True}],
}


def get_model_caller(name=model_names.vgg16):
    caller, kwargs=available_models[name]

    def model_caller():
        model=caller(kwargs).eval().to(device)
        model.name=name  # save its name
        closeParamGrad(model)  # not learn
        closeInplace(model)  # not inplace
        return model
    return model_caller


def get_model(name='vgg16'):
    model = get_model_caller(name)()
    return model


def closeParamGrad(model):
    for para in model.parameters():
        para.requires_grad=False


def closeInplace(model):
    for m in model.modules():
        if hasattr(m, 'inplace'):
            m.inplace=False


def auto_find_layer_index(model, layer=-1):
    # index used by lrp
    # only for vgg16(sequential like).
    # layer is int or str
    # layer 0 is input layer, then features follows that features_0 is layer 1
    if layer is None:
        layer = -1
    index = layer % (1 + len(model.features))  # 0 is input layer
    return index


INTPUT_LAYER = 'input_layer'


# get layer name strings
def decode_stages(model, stages=(0, 1, 2, 3, 4, 5)):
    if not isinstance(stages, (list, tuple)):
        stages = (stages,)
    if isinstance(model, VGG):
        layer_names = ['input_layer'] + [('features', i) for i in (4, 9, 16, 23, 30)]
    elif isinstance(model, ResNet):
        layer_names = ['input_layer', 'maxpool'] + [(f'layer{i}', -1) for i in (1, 2, 3, 4)]
    elif isinstance(model, GoogLeNet):
        layer_names = ['input_layer', 'maxpool1', 'maxpool2', 'inception3b', 'inception4e', 'inception5b']
    elif isinstance(model, AlexNet):
        layer_names = ['input_layer'] + [('features', i) for i in (0, 2, 5, 11, 12)]
    elif isinstance(model, VisionTransformer):
        layer_names = ['input_layer', 'conv_proj'] + [('encoder', 'layers', i) for i in (0, 3, 6, 9)]
    else:
        raise Exception(f'{model.__class__} is not available model type')
    return [layer_names[stage] for stage in stages]


# get real in model by name.
def findLayerByName(model, layer_name=(None,)):
    if not isinstance(layer_name, (tuple, list)):
        layer_name = (layer_name,)
    if layer_name[0] == INTPUT_LAYER:
        return model  # model itself
    layer = model
    for l in layer_name:
        if isinstance(l, int):  # for sequential
            layer = layer[l]
        elif isinstance(l, str) and hasattr(layer, l):  # for named child
            layer = layer.__getattr__(l)
        else:
            raise Exception(f'no layer:{layer_name} in model:{model}')
    return layer


def saving_activation(obj, in_mode=False):
    def wrapper(module, inputs, output):
        if in_mode:
            obj.activation = inputs[0].clone()
        else:
            obj.activation = output.clone()

    return wrapper


def saving_gradient(obj, in_mode=False):
    def wrapper(module, grad_inputs, grad_outputs):
        if in_mode:
            obj.gradient = grad_inputs[0].clone()
        else:
            obj.gradient = grad_outputs[0].clone()

    return wrapper


def saving_both(layer, in_mode=False):  # bin-way
    hooks = []
    hooks.append(layer.register_forward_hook(saving_activation(layer, in_mode)))
    hooks.append(layer.register_full_backward_hook(saving_gradient(layer, in_mode)))
    return hooks


# save activations and gradients in corresponding layers, automatically detect input-layer.
def auto_hook(model, layer_names):
    layers = []
    model.hooks = []
    if not isinstance(layer_names, (tuple, list)):
        layer_names = (layer_names,)
    for layer_name in layer_names:
        if layer_name == INTPUT_LAYER:  # fake layer
            layers.append(model)
            model.hooks.extend(saving_both(model, True))
        else:  # real layer
            layer = findLayerByName(model, layer_name)
            layers.append(layer)
            model.hooks.extend(saving_both(layer))
    return layers


def clearHooks(model):
    for h in model.hooks:
        h.remove()
    model.hooks.clear()


def forward_hook(obj, module, input, output):
    obj.activation = output.clone().detach()


def backward_hook(obj, module, grad_input, grad_output):
    obj.gradient = grad_output[0].clone().detach()


# deprecated
def hookLayerByName(obj, model, layer_name=(None,)):
    # obj: save a,g to where?
    if not hasattr(obj, 'hooks'):
        obj.hooks = []

        def clearHooks(obj):
            for h in obj.hooks:
                h.remove()
            obj.hooks.clear()

        obj.clearHooks = clearHooks
    if layer_name == 'input_layer':
        obj.hooks.append(model.register_forward_hook(saving_activation(obj, True)))
        obj.hooks.append(model.register_full_backward_hook(saving_gradient(model, True)))
    else:
        layer = findLayerByName(model, layer_name)
        obj.hooks.append(layer.register_forward_hook(saving_activation(obj)))
        obj.hooks.append(layer.register_full_backward_hook(saving_gradient(obj)))


def relevanceFindByName(model, layer_name=(None,)):
    # compatible for input layer
    if layer_name == 'input_layer' or layer_name[0] == 'input_layer':
        return model.x.diff(dim=0) * model.g
    else:
        layer = findLayerByName(model, layer_name)
        hm = layer.y.diff(dim=0) * layer.g
        return hm
