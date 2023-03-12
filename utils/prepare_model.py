import torchvision as tv

device = 'cuda'

def get_model(name='vgg16',device=device):
    models = {
        "vgg16": lambda: tv.models.vgg16(weights=tv.models.VGG16_Weights.DEFAULT).eval().to(device),
        "alexnet": lambda: tv.models.alexnet(weights=tv.models.AlexNet_Weights.DEFAULT).eval().to(device),
        "resnet18": lambda: tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT).eval().to(device),
        "resnet50": lambda: tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT).eval().to(device),
    }
    return models[name]()


# target_layer used for hook-based method
# layer must be str
def auto_find_layer(model, layer='features_-1'):
    if layer is None:
        layer = 'features_-1'
    if isinstance(layer, str):
        s = layer.split('_')
        # 3 cases: given features+?, classifier+?, ?
        if len(s) == 1:  # if no suffix, it will be -1
            s.append('-1')
        if s[0] == 'features':
            sub_module = model.features
        elif s[0] == 'classifier':
            sub_module = model.classifier
        else:
            sub_module = model.features  # raw num is features_num
            s[1] = s[0]
        assert len(s) == 2
        off_set = int(s[1]) % len(sub_module)
        target_layer = list(sub_module)[off_set]
    else:
        raise Exception()
    return target_layer

# index used by lrp model
# layer is int or str
# index 0 is x layer
def auto_find_layer_index(model, layer=-1):
    # if given int ,search model[input,features]
    # if given str ,search specific module
    # that means differed by 1 among two types
    if layer is None:
        layer = 'features_-1'
    if isinstance(layer, str):
        s = layer.split('_')
        index = 1  # str can not access x layer.
        # 3 cases: given features+?, classifier+?, ?
        if len(s) == 1:
            s.append('-1')
        if s[0] == 'features':  # features_n
            sub_module = model.features
        elif s[0] == 'classifier':  # classifier_n
            sub_module = model.classifier
            index += len(model.features) + 1
        else:  # n
            sub_module = model.features
            s[1] = s[0]
        assert len(s) == 2
        off_set = int(s[1]) % len(sub_module)
        index += off_set
    elif isinstance(layer, int):
        index = layer % (1 + len(model.features))
    else:
        raise Exception()
    return index
# #---test
# model = get_model()
# print(auto_find_layer(model, -1))
# print(auto_find_layer(model, f'features_-1'))
# for i in range(30):
#     print(auto_find_layer(model, i))
#     print(auto_find_layer(model, f'features_{i}'))
