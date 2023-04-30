import torchvision as tv

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


# index used by lrp
# only for vgg16(sequential like).
# layer is int or str
# index 0 is x layer
def auto_find_layer_index(model, layer=-1):
    # if given int ,search model[input,features]
    # if given str ,search specific module
    # that means differed by 1 among two types
    if layer is None:
        layer = -1
    if isinstance(layer, str):
        layer = int(layer)
    index = layer % (1 + len(model.features))  # 0 is input layer
    return index


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
        obj.hooks.append(model.register_forward_hook(lambda *args: save_act_in(obj, *args)))
        obj.hooks.append(model.register_full_backward_hook(lambda *args: save_grad_in(obj, *args)))
    else:
        layer = findLayerByName(model, layer_name)
        obj.hooks.append(layer.register_forward_hook(lambda *args: forward_hook(obj, *args)))
        obj.hooks.append(layer.register_full_backward_hook(lambda *args: backward_hook(obj, *args)))


def relevanceFindByName(model, layer_name=(None,)):
    # compatible for input layer
    if layer_name == 'input_layer' or layer_name[0] == 'input_layer':
        return model.x.diff(dim=0) * model.gx
    else:
        layer = findLayerByName(model, layer_name)
        return layer.y.diff(dim=0) * layer.g