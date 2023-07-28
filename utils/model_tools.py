import torchvision as tv
from torchvision.models import VGG, AlexNet, ResNet, GoogLeNet, VisionTransformer

device = 'cuda'
avaiable_models = {
    "vgg16": lambda: tv.models.vgg16(weights=tv.models.VGG16_Weights.DEFAULT).eval().to(device),
    "alexnet": lambda: tv.models.alexnet(weights=tv.models.AlexNet_Weights.DEFAULT).eval().to(device),
    "resnet18": lambda: tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT).eval().to(device),
    "resnet34": lambda: tv.models.resnet34(weights=tv.models.ResNet34_Weights.DEFAULT).eval().to(device),
    "resnet50": lambda: tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT).eval().to(device),
    "googlenet": lambda: tv.models.googlenet(weights=tv.models.GoogLeNet_Weights.DEFAULT).eval().to(device),
    "vit": lambda: tv.models.vit_b_16(weights=tv.models.ViT_B_16_Weights.DEFAULT).eval().to(device),
}


def get_model(name='vgg16'):
    return avaiable_models[name]()


def auto_find_layer_index(model, layer=-1):
    # index used by lrp
    # only for vgg16(sequential like).
    # layer is int or str
    # layer 0 is input layer, then features follows that features_0 is layer 1
    if layer is None:
        layer = -1
    index = layer % (1 + len(model.features))  # 0 is input layer
    return index


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
        layer_names = ['input_layer', 'input_layer'] + [('features', i) for i in (0, 2, 5, 12)] # alex only exists last 4 stages
    elif isinstance(model, VisionTransformer):
        layer_names = ['input_layer', 'conv_proj'] + [('encoder','layers', i) for i in (0, 3, 6, 9)]
    else:
        raise Exception(f'{model.__class__} is not available model type')
    return [layer_names[stage] for stage in stages]


def findLayerByName(model, layer_name=(None,)):
    layer = model
    if not isinstance(layer_name, (tuple, list)):
        layer_name = (layer_name,)
    for l in layer_name:
        if isinstance(l, int):  # for sequential
            layer = layer[l]
        elif isinstance(l, str) and hasattr(layer, l):  # for named child
            layer = layer.__getattr__(l)
        else:
            raise Exception(f'no layer:{layer_name} in model:{model}')
    return layer


def forward_hook(obj, module, input, output):
    obj.activation = output.clone().detach()


def backward_hook(obj, module, grad_input, grad_output):
    obj.gradient = grad_output[0].clone().detach()


def save_act_in(obj, module, input, output):
    obj.activation = input[0].clone().detach()


def save_grad_in(obj, module, grad_input, grad_output):
    obj.gradient = grad_input[0].clone().detach()


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
        obj.hooks.append(model.register_forward_hook(lambda *args, obj=obj: save_act_in(obj, *args)))
        obj.hooks.append(model.register_full_backward_hook(lambda *args, obj=obj: save_grad_in(obj, *args)))
    else:
        layer = findLayerByName(model, layer_name)
        obj.hooks.append(layer.register_forward_hook(lambda *args, obj=obj: forward_hook(obj, *args)))
        obj.hooks.append(layer.register_full_backward_hook(lambda *args, obj=obj: backward_hook(obj, *args)))


def relevanceFindByName(model, layer_name=(None,)):
    # compatible for input layer
    if layer_name == 'input_layer' or layer_name[0] == 'input_layer':
        return model.x.diff(dim=0) * model.gx
    else:
        layer = findLayerByName(model, layer_name)
        return layer.y.diff(dim=0) * layer.g