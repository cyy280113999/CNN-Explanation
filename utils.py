import os
import random
import sys
import time
from itertools import product
import json

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
import timm
import torch
import torch.nn.functional as nf
import torch.utils.data as TD
import torchvision
import torchvision as tv
import torchvision.transforms.functional as vf
from PIL import Image
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout, QPushButton, QLineEdit, QApplication, QComboBox, \
    QFileDialog, QListWidget, QAbstractItemView, QMessageBox
from torchvision.models import VGG, AlexNet, ResNet, GoogLeNet, VisionTransformer

device = 'cuda'


# ==========================================================================
# ==================  image/tensor tools   =======================
# ==========================================================================


def generate_abs_filename(this, fn):
    current_file_path = os.path.abspath(this)
    current_directory = os.path.dirname(current_file_path)
    file_to_read = os.path.join(current_directory, fn)
    return file_to_read


# ========== pil image loading
def pilOpen(filename):
    return Image.open(filename).convert('RGB')


# =========== image process
toRRC = torchvision.transforms.RandomResizedCrop(224, scale=(0.25, 1), ratio=(1, 1))

# image PIL -> tensor (c,h,w)
toTensor = torchvision.transforms.ToTensor()

toTensorS224 = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    toTensor,
])


class Pad224(torch.nn.Module):
    def forward(self, x):
        size = 224
        _, h, w = vf.get_dimensions(x)
        short, long = (w, h) if w <= h else (h, w)
        new_short, new_long = int(size * short / long), size
        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
        x = vf.resize(x, size=[new_h, new_w])
        x = vf.center_crop(x, [size, size])
        return x


toTensorPad224 = torchvision.transforms.Compose([
    Pad224(),
    toTensor,
])

# tensor (c,h,w) to PIL
toPIL = torchvision.transforms.ToPILImage()


# ======== plot functions
# tensor with (1,c,h,w) -> numpy with (h,w,c)
def toPlot(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        # case `(N, C, H, W)`
        if len(x.shape) == 4:
            assert x.shape[0] == 1
            x = x.squeeze(0)
        # case `(H, W)`
        if len(x.shape) == 2:
            x = x.reshape((1,) + x.shape)
        if len(x.shape) != 3:
            raise TypeError('mismatch dimension')
        # case `(C, H, W)`
        return x.transpose(1, 2, 0)  # hwc
    else:
        raise TypeError(f'Plot Type is unavailable for {type(x)}')


# =============== image standardize  ===========================
ImgntMean = [0.485, 0.456, 0.406]
ImgntStd = [0.229, 0.224, 0.225]
ImgntMeanTensor = torch.tensor(ImgntMean).reshape(1, -1, 1, 1)
ImgntStdTensor = torch.tensor(ImgntStd).reshape(1, -1, 1, 1)


class Normalizer:
    def __init__(self):
        self.cpus = (ImgntMeanTensor, ImgntStdTensor)
        self.cudas = (ImgntMeanTensor.cuda(), ImgntStdTensor.cuda())
        self.normalize_cpu = torchvision.transforms.Normalize(*self.cpus)
        self.normalize_cuda = torchvision.transforms.Normalize(*self.cudas)

    def toStd(self, x):
        if x.device == 'cuda':
            return self.normalize_cuda(x)
        else:
            return self.normalize_cpu(x)

    def invStd(self, x):
        if x.device == 'cuda':
            return x * self.cudas[1] + self.cudas[0]
        else:
            return x * self.cpus[1] + self.cpus[0]


normalizer = Normalizer()

# that is compatible for both cpu and cuda
toStd = normalizer.toStd

invStd = normalizer.invStd


# ============ std image loading
# image for raw (PIL,numpy) image. x for standardized tensor
def get_image_x(filename='cat_dog_243_282.png', image_folder='input_images/', device=device):
    # require folder , pure filename
    filename = image_folder + filename
    img_PIL = pilOpen(filename)
    img_tensor = toTensorS224(img_PIL).unsqueeze(0)
    img_tensor = toStd(img_tensor).to(device)
    return img_tensor


# =========== dataset
# ImageNet loading
if sys.platform == 'win32':
    imageNetDefaultDir = r'F:/DataSet/imagenet/'
elif sys.platform == 'linux':
    imageNetDefaultDir = r'/home/dell/datasets/imgnt/'
else:
    imageNetDefaultDir = None
imageNetSplits = {
    'train': 'train/',
    'val': 'val/',
}

default_transform = torchvision.transforms.Compose([
    toTensorS224,
    toStd
])

show_transform = torchvision.transforms.Compose([
    toTensorPad224,
    toStd
])

_classes = None


def label_translate(i):
    global _classes
    if _classes is None:
        with open('imagenet_class_index.json') as f:
            _classes = json.load(f)
            _classes = {int(i): v[-1] for i, v in _classes.items()}
    return _classes[i]


def loadImageNetClasses(path=''):
    import json
    filename = path + 'imagenet_class_index.json'
    with open(filename) as f:
        c = json.load(f)
        c = {int(i): v[-1] for i, v in c.items()}
        return c


def getImageNet(split, transform=default_transform):
    try:
        return torchvision.datasets.ImageNet(root=imageNetDefaultDir,
                                             split=split,
                                             transform=transform)
    except Exception as e:
        print(e)
        return None


# =============== image dataset with one folder
import os
from torchvision.datasets.folder import ImageFolder, default_loader


def find_classes(directory):
    classes = (0,)
    class_to_idx = {"": 0}
    return classes, class_to_idx

# 文件夹中只有图片，生成一个无类别的数据集，类别为0
class OnlyImages(ImageFolder):
    def find_classes(self, directory: str):
        return find_classes(directory)


# ==========================================================================
# ==================  model tools   =======================
# ==========================================================================


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
    l = set()
    for k, v in model_names.__dict__:
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
    caller, kwargs = available_models[name]

    def model_caller():
        model = caller(**kwargs).eval().to(device)
        model.name = name  # save its name
        closeParamGrad(model)  # not learn
        closeInplace(model)  # not inplace
        return model

    return model_caller


def get_model(name='vgg16'):
    model = get_model_caller(name)()
    return model


def closeParamGrad(model):
    for para in model.parameters():
        para.requires_grad = False


def closeInplace(model):
    for m in model.modules():
        if hasattr(m, 'inplace'):
            m.inplace = False


# ============================= find layers  ===============================

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


# ============================ hook layers ===================
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
            model.hooks.extend(saving_both(model, in_mode=True))
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


# ==========================================================================
# ==================  utilities   =======================
# ==========================================================================


# make path (recursively)
def mkp(p):
    d = os.path.dirname(p)
    if not os.path.exists(d):
        os.makedirs(d)


class EmptyObject:
    pass


# running cost
class RunningCost:
    def __init__(self, stage_count=5):
        self.stage_count = stage_count
        self.running_cost = [None for i in enumerate(range(self.stage_count + 1))]
        self.hint = [None for i in enumerate(range(self.stage_count))]
        self.position = 0

    def tic(self, hint=None):
        if self.position < self.stage_count:
            t = time.time()
            self.running_cost[self.position] = t
            self.hint[self.position] = hint
            self.position += 1

    def cost(self):
        print('-' * 20)
        for stage_, (i, j) in enumerate(zip(self.running_cost, self.running_cost[1:])):
            if j is not None:
                if self.hint[stage_ + 1] is not None:
                    print(f'stage {self.hint[stage_ + 1]} cost time: {j - i}')
                else:
                    print(f'stage {stage_ + 1} cost time: {j - i}')

        print('-' * 20)


# ============= data debug
def tensor_info(tensor, print_info=True):
    methods = {'min': torch.min,
               'max': torch.max,
               'mean': torch.mean,
               'std': torch.std}
    data = []
    for n, m in methods.items():
        data.append((n, m(tensor).item()))
    if print_info:
        print(data)
    else:
        return data


# show tensor
def show_heatmap(x):
    glw = pg.GraphicsLayoutWidget()
    pi = glw.addPlot()
    x = toPlot(heatmapNormalizeR(x.sum(1, True)))
    ii = pg.ImageItem(x, levels=[-1, 1], lut=lrp_lut)
    pi.addItem(ii)
    plotItemDefaultConfig(pi)
    glw.show()
    pg.exec()


def show_image(tensor):
    glw = pg.GraphicsLayoutWidget()
    pim = glw.addPlot()
    ii = pg.ImageItem(toPlot(tensor), levels=[0, 1])
    pim.addItem(ii)
    plotItemDefaultConfig(pim)
    glw.show()
    pg.exec()


def save_tensor(tensor, path):
    mkp(path)
    torch.save(tensor, path)


def save_image(tensor, path):
    mkp(path)
    ii = pg.ImageItem(toPlot(tensor), levels=[0, 1])
    ii.save(path)


def save_heatmap(tensor, path):
    mkp(path)
    tensor = heatmapNormalizeR(tensor.sum(1, True))
    ii = pg.ImageItem(toPlot(tensor), levels=[-1, 1], lut=lrp_lut)
    ii.save(path)


def showVRAM():
    print(torch.cuda.memory_summary())
    torch.cuda.empty_cache()


def histogram(x):
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.device.type == 'cuda':
            x = x.cpu()
    # histogram
    a, b = x.histogram()
    a = a.numpy()
    width = b[1] - b[0]
    b += width
    b = b[:-1].numpy()
    plt.bar(b, a, width)
    plt.show()


# ==========================================================================
# ==================  heatmap process   =======================
# ==========================================================================


def heatmapNormalizeR(heatmap):
    M = heatmap.abs().max()
    heatmap = heatmap / M
    heatmap = torch.nan_to_num(heatmap)
    return heatmap


def heatmapNormalizeR_ForEach(heatmap):
    M = torch.max_pool2d(heatmap.abs(), kernel_size=heatmap.shape[2:])
    heatmap = heatmap / M
    heatmap = torch.nan_to_num(heatmap)
    return heatmap


def heatmapNR2P(heatmap):
    return heatmap / 2 + 0.5


def interpolate_to_imgsize(heatmap, size=(224, 224)):  # only for heatmap
    return heatmapNormalizeR(nf.interpolate(heatmap.sum(1, True), size=size, mode='bilinear'))


def multi_interpolate(heatmaps):
    return heatmapNormalizeR(sum(interpolate_to_imgsize(x) for x in heatmaps))



# ============ drawing
# image save
lrp_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
lrp_cmap[:, 0:3] *= 0.85
lrp_cmap = matplotlib.colors.ListedColormap(lrp_cmap)

lrp_cmap_gl = pg.colormap.get('seismic', source='matplotlib')
# lrp_cmap_gl.color[:,0:3]*=0.8
# lrp_cmap_gl.color[2,3]=0.5
lrp_lut = lrp_cmap_gl.getLookupTable(start=0, stop=1, nPts=256)


# ========== pyqtgraph
def pyqtgraphDefaultConfig():
    pg.setConfigOptions(**{'imageAxisOrder': 'row-major',
                           'background': 'w',
                           'foreground': 'k',
                           # 'useNumba': True,
                           # 'useCupy': True,
                           })


pyqtgraphDefaultConfig()


def plotItemDefaultConfig(p):
    # this for showing image
    p.showAxes(False)
    p.invertY(True)
    p.vb.setAspectLocked(True)


# ==========================================================================
# ==================  heatmap mask   =======================
# ==========================================================================


def binarize(mask, sparsity=0.5):
    x = mask.flatten()
    n = len(x)
    ascending = x.sort()[0]
    value = ascending[int(sparsity * n)]
    return 1.0 * (mask >= value)


def bin_drop_last(mask, sparsity=0.5):
    x = mask.flatten()
    n = len(x)
    ascending = x.sort()[0]
    value = ascending[int(sparsity * n)]
    return 1.0 * (mask >= value)


def bin_keep_last(mask, sparsity=0.5):
    x = mask.flatten()
    n = len(x)
    ascending = x.sort()[0]
    value = ascending[int(sparsity * n)]
    return 1.0 * (mask <= value)


def bin_keep_first(mask, sparsity=0.5):
    x = mask.flatten()
    n = len(x)
    ascending = x.sort()[0]
    value = ascending[int((1 - sparsity) * n)]
    return 1.0 * (mask >= value)


def bin_drop_first(mask, sparsity=0.5):
    x = mask.flatten()
    n = len(x)
    ascending = x.sort()[0]
    value = ascending[int((1 - sparsity) * n)]
    return 1.0 * (mask <= value)


def positize(mask):
    return 1.0 * (mask >= 0)


def binarize_mul_noisy_n(mask, sparsity=0.5, top=True, std=0.1):
    x = mask.flatten()
    threshold = x.sort()[0][int(sparsity * len(x))]
    if top:
        mask = 1.0 * (mask <= threshold) + std * torch.randn_like(mask) * (mask > threshold)
    else:
        mask = 1.0 * (mask >= threshold) + std * torch.randn_like(mask) * (mask < threshold)
    return mask


def binarize_add_noisy_n(mask, sparsity=0.5, top=True, std=0.1):
    x = mask.flatten()
    threshold = x.sort()[0][int(sparsity * len(x))]
    if top:
        mask = std * torch.randn_like(mask) * (mask > threshold)
    else:
        mask = std * torch.randn_like(mask) * (mask < threshold)
    return mask


def maximalLoc(heatmap2d, top=True):
    if len(heatmap2d.shape) == 4:
        heatmap2d = heatmap2d.squeeze(0)
    if len(heatmap2d.shape) == 3:
        heatmap2d = heatmap2d.sum(0)
    assert len(heatmap2d.shape) == 2
    h, w = heatmap2d.shape
    f = heatmap2d.flatten()
    loc = f.sort(descending=top)[1][0].item()
    x = loc // h
    y = loc - (x * h)
    return x, y


def patch(heatmap2d, loc, r=1):
    if len(heatmap2d.shape) == 4:
        heatmap2d = heatmap2d.squeeze(0)
    if len(heatmap2d.shape) == 3:
        heatmap2d = heatmap2d.sum(0)
    h, w = heatmap2d.shape
    x, y = loc
    xL = max(0, x - r)
    xH = min(h, x + r)
    yL = max(0, y - r)
    yH = min(w, y + r)
    patched = torch.ones_like(heatmap2d)
    patched[xL:xH + 1, yL:yH + 1] = 0
    return patched


def maximalPatch(heatmap2d, top=True, r=1):
    if len(heatmap2d.shape) == 4:
        heatmap2d = heatmap2d.squeeze(0)
    if len(heatmap2d.shape) == 3:
        heatmap2d = heatmap2d.sum(0)
    assert len(heatmap2d.shape) == 2
    h, w = heatmap2d.shape
    f = heatmap2d.flatten()
    loc = f.sort(descending=top)[1][0].item()
    x = loc // h
    y = loc - (x * h)

    xL = max(0, x - r)
    xH = min(h, x + r)
    yL = max(0, y - r)
    yH = min(w, y + r)
    patched = torch.ones_like(heatmap2d)
    patched[xL:xH + 1, yL:yH + 1] = 0
    return patched


def cornerMask(mask2d, r=1):
    if len(mask2d.shape) == 4:
        mask2d = mask2d.squeeze(0)
    if len(mask2d.shape) == 3:
        mask2d = mask2d.sum(0)
    assert len(mask2d.shape) == 2
    h, w = mask2d.shape
    patched = torch.ones_like(mask2d)
    for x, y in product((0, h - 1), (0, w - 1)):
        xL = max(0, x - r)
        xH = min(h, x + r)
        yL = max(0, y - r)
        yH = min(w, y + r)
        patched[xL:xH + 1, yL:yH + 1] = 0
    return patched


# ==========================================================================
# ==================  window tools   =======================
# ==========================================================================


qapp = None
loop_flag = None


class TippedWidget(QWidget):
    def __init__(self, tip="Empty Tip", widget=None):
        super(TippedWidget, self).__init__()
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        main_layout.addWidget(QLabel(tip))
        if widget is None:
            raise Exception("Must given widget.")
        self.widget = widget
        main_layout.addWidget(self.widget)

    def __getitem__(self, item):
        return self.widget.__getitem__(item)


class DictComboBox(QComboBox):
    def __init__(self, combo_dict, ShapeMode=1):
        super().__init__()
        if ShapeMode == 0:
            for k, v in combo_dict.items():
                self.addItem(k)
        elif ShapeMode == 1:
            temp = QStandardItemModel()
            for key in combo_dict:
                temp2 = QStandardItem(key)
                temp2.setData(key)  # , Qt.ToolTipRole
                temp2.setSizeHint(QSize(200, 40))
                temp.appendRow(temp2)
            self.setModel(temp)
        self.setCurrentIndex(0)
        self.setMinimumHeight(40)


class ListComboBox(QComboBox):
    def __init__(self, l, ShapeMode=1):
        super().__init__()
        if ShapeMode == 0:
            for x in l:
                self.addItem(x)
        elif ShapeMode == 1:
            temp = QStandardItemModel()
            for x in l:
                temp2 = QStandardItem(x)
                temp2.setData(x)  # , Qt.ToolTipRole
                temp2.setSizeHint(QSize(200, 40))
                temp.appendRow(temp2)
            self.setModel(temp)
        self.setCurrentIndex(0)
        self.setMinimumHeight(40)


class ListWidget(QListWidget):
    def __init__(self, l, ShapeMode=1):
        super().__init__()
        self.addItems(l)
        self.setCurrentRow(0)
        # self.method_list.setSelectionMode(QAbstractItemView.MultiSelection)
        # self.method_list.clear()


class ImageCanvas(QWidget):
    def __init__(self):
        super(ImageCanvas, self).__init__()
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        self.pglw: pg.GraphicsLayout = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.pglw)

    def showImage(self, img, levels=(0, 1), lut=None):
        assert isinstance(img, np.ndarray)
        self.pglw.clear()
        pi: pg.PlotItem = self.pglw.addPlot()
        plotItemDefaultConfig(pi)
        ii = pg.ImageItem(img, levels=levels, lut=lut)
        pi.addItem(ii)

    def showImages(self, imgs, size=(1, 1), levels=(0, 1), lut=None):
        for i, img in enumerate(imgs):
            row = i // size[0]
            col = i % size[0]
            pi: pg.PlotItem = self.pglw.addPlot(row=row, col=col)
            plotItemDefaultConfig(pi)
            ii = pg.ImageItem(img, levels=levels, lut=lut)
            pi.addItem(ii)


class BaseDatasetTravellingVisualizer(QWidget):
    """
    if it is not inherited,  use imageChangeCallBack=your_call_back instead.
    """

    def __init__(self, dataset, AddCanvas=True, imageChangeCallBack=None):
        super().__init__()
        self.dataSet = dataset
        self.imageSelector = DatasetTraveller(self.dataSet)
        self.raw_inputs = None
        self.initUI()
        # canvas
        if AddCanvas:
            self.imageCanvas = ImageCanvas()
            self.main_layout.addWidget(self.imageCanvas)
        if imageChangeCallBack is not None:
            self.imageChange = lambda: imageChangeCallBack(self)
        self.getImage()

    def initUI(self):
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        hlayout = QHBoxLayout()  # add row
        self.dataSetInfo = QLabel()
        self.dataSetInfo.setText(f"Images Index From 0 to {len(self.dataSet) - 1}")
        hlayout.addWidget(self.dataSetInfo)

        self.back = QPushButton("Back")
        self.next = QPushButton("Next")
        self.randbtn = QPushButton("Rand")
        self.index = QLineEdit("0")

        hlayout.addWidget(self.back)
        hlayout.addWidget(self.next)
        hlayout.addWidget(self.randbtn)
        hlayout.addWidget(self.index)
        self.main_layout.addLayout(hlayout)
        self.back.setMinimumHeight(40)
        self.next.setMinimumHeight(40)
        self.randbtn.setMinimumHeight(40)
        self.index.setMinimumHeight(40)
        self.index.setMaximumWidth(80)
        self.index.setMaxLength(8)

        self.imgInfo = QLabel("Image Info:")
        self.main_layout.addWidget(self.imgInfo)
        self.next.clicked.connect(self.indexNext)
        self.back.clicked.connect(self.indexBack)
        self.randbtn.clicked.connect(self.indexRand)
        self.index.returnPressed.connect(self.getImage)

    def indexNext(self):
        self.raw_inputs = self.imageSelector.next()
        self.index.setText(str(self.imageSelector.index))
        self.imageChange()

    def indexBack(self):
        self.raw_inputs = self.imageSelector.back()
        self.index.setText(str(self.imageSelector.index))
        self.imageChange()

    def indexRand(self):
        self.raw_inputs = self.imageSelector.rand()
        self.index.setText(str(self.imageSelector.index))
        self.imageChange()

    def getImage(self):
        self.raw_inputs = self.imageSelector.get(int(self.index.text()))
        self.index.setText(str(self.imageSelector.index))
        self.imageChange()

    def imageChange(self):
        # img, cls = self.raw_input
        # self.imageCanvas.showImage(np.array(img))
        raise NotImplementedError()


def windowMain(WindowClass):
    qapp = QApplication.instance()
    loop_flag = True
    if qapp is None:
        qapp = QApplication(sys.argv)
    else:
        loop_flag = False
    mw = WindowClass()
    mw.show()
    # if loop_flag:
    #     sys.exit(qapp.exec_())


def create_qapp():
    global qapp
    if qapp is None:
        qapp = QApplication(sys.argv)


def loop_qapp():
    global loop_flag
    if not loop_flag:
        loop_flag = True
        qapp = QApplication.instance()
        sys.exit(qapp.exec_())


# dataset window
class DatasetTraveller:
    def __init__(self, dataset):
        super().__init__()
        self.dataSet = dataset
        self.dataSetLen = len(dataset)
        self.img = None
        self.index = 0
        import time
        np.random.seed(int(time.time()))
        self.check = lambda x: x % len(self.dataSet)

    def get(self, i=None):
        if i is not None:
            self.index = self.check(i)
        return self.dataSet[self.index]

    def next(self):
        self.index = self.check(self.index + 1)
        return self.dataSet[self.index]

    def back(self):
        self.index = self.check(self.index - 1)
        return self.dataSet[self.index]

    def rand(self):
        i = np.random.randint(0, self.dataSetLen - 1, (1,))[0]
        self.index = self.check(i)
        return self.dataSet[self.index]


class SubSetFromIndices(TD.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)


class SingleImageLoader(QWidget):
    img = None

    def __init__(self, sendCallBack=None):
        super().__init__()
        self.init_ui()
        self.open_btn.clicked.connect(self.open)
        self.link(sendCallBack)

    def init_ui(self):
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.img_info = QLabel()
        self.set_info()
        self.main_layout.addWidget(self.img_info)
        self.open_btn = QPushButton('Open')
        self.main_layout.addWidget(self.open_btn)

    def set_info(self, p='None'):
        self.img_info.setText(f'Image Info: {p}')

    def open(self):
        filename_long, f_type = QFileDialog.getOpenFileName(directory="./")
        if filename_long:
            self.img = pilOpen(filename_long)
            self.img = toTensor(self.img).unsqueeze(0)
            self.set_info(os.path.basename(filename_long))
            if self.send is not None:
                self.send(self.img)
        else:
            self.img = None
            self.set_info()
            pass

    def link(self, sendCallBack):
        self.send = None
        if sendCallBack is not None:
            self.send = sendCallBack


class FoldImageLoader(QWidget):
    dataSet = None
    index = 0
    img = None

    def __init__(self, sendCallBack=None):
        super().__init__()
        self.initUI()
        self.open_btn.clicked.connect(self.open)
        self.next.clicked.connect(self.indexNext)
        self.back.clicked.connect(self.indexBack)
        self.randbtn.clicked.connect(self.indexRand)
        self.index_editer.returnPressed.connect(self.parse_index)
        self.link(sendCallBack)

    def initUI(self):
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.datasetInfo = QLabel()
        self.set_dataset_info()
        self.main_layout.addWidget(self.datasetInfo)
        self.imgInfo = QLabel()
        self.set_image_info()
        self.main_layout.addWidget(self.imgInfo)
        hlayout = QHBoxLayout()  # add row
        self.open_btn = QPushButton("Open")
        self.back = QPushButton("Back")
        self.next = QPushButton("Next")
        self.randbtn = QPushButton("Rand")
        self.index_editer = QLineEdit("0")
        hlayout.addWidget(self.open_btn)
        hlayout.addWidget(self.back)
        hlayout.addWidget(self.next)
        hlayout.addWidget(self.randbtn)
        hlayout.addWidget(self.index_editer)
        self.main_layout.addLayout(hlayout)
        self.open_btn.setMinimumHeight(40)
        self.back.setMinimumHeight(40)
        self.next.setMinimumHeight(40)
        self.randbtn.setMinimumHeight(40)
        self.index_editer.setMinimumHeight(40)
        self.index_editer.setMaximumWidth(80)
        self.index_editer.setMaxLength(8)
        self.btns = [self.back, self.next, self.randbtn, self.index_editer]
        self.show_btns(False)

    def set_dataset_info(self):
        if self.dataSet is None:
            self.datasetInfo.setText(f"Please open.")
        elif len(self.dataSet) == 0:
            self.datasetInfo.setText(f"No image in folder.")
        else:
            self.datasetInfo.setText(f"Images Index From 0 to {len(self.dataSet) - 1}.")

    def set_image_info(self, p='None'):
        self.imgInfo.setText(f'Image Info: {p}')

    def show_btns(self, flag=True):
        if flag:
            func = lambda x: x.show()
        else:
            func = lambda x: x.hide()
        for b in self.btns:
            func(b)

    def getImage(self):
        self.img, c = self.dataSet[self.index]
        p, _ = self.dataSet.samples[self.index]
        self.set_image_info(os.path.basename(p))
        self.img = toTensor(self.img).unsqueeze(0)
        if self.send is not None:
            self.send(self.img)

    def set_index(self, i):
        i = i % len(self.dataSet)
        self.index = i
        self.index_editer.setText(str(i))
        self.getImage()

    def parse_index(self):
        t = self.index_editer.text()
        try:
            i = int(t)
            self.set_index(i)
        except Exception() as e:
            pass

    def open(self):
        directory = QFileDialog.getExistingDirectory(directory="./")
        if directory:
            self.dataSet = OnlyImages(directory)
            self.set_dataset_info()
            if len(self.dataSet) == 0:
                self.show_btns(False)
            else:
                self.show_btns()
                self.set_index(0)

    def indexNext(self):
        self.set_index(self.index + 1)

    def indexBack(self):
        self.set_index(self.index - 1)

    def indexRand(self):
        self.set_index(random.randint(0, len(self.dataSet) - 1) + 1)

    def link(self, sendCallBack):
        self.send = None
        if sendCallBack is not None:
            self.send = sendCallBack


class TorchDatesetLoader(QWidget):
    dataSet = None
    index = 0
    img = None

    def __init__(self, sendCallBack=None):
        super().__init__()
        self.initUI()
        self.next.clicked.connect(self.indexNext)
        self.back.clicked.connect(self.indexBack)
        self.randbtn.clicked.connect(self.indexRand)
        self.index_editer.returnPressed.connect(self.parse_index)
        self.link(sendCallBack)

    def initUI(self):
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.datasetInfo = QLabel()
        self.set_dataset_info()
        self.main_layout.addWidget(self.datasetInfo)
        self.imgInfo = QLabel()
        self.set_image_info()
        self.main_layout.addWidget(self.imgInfo)
        hlayout = QHBoxLayout()  # add row
        self.back = QPushButton("Back")
        self.next = QPushButton("Next")
        self.randbtn = QPushButton("Rand")
        self.index_editer = QLineEdit("0")
        hlayout.addWidget(self.back)
        hlayout.addWidget(self.next)
        hlayout.addWidget(self.randbtn)
        hlayout.addWidget(self.index_editer)
        self.main_layout.addLayout(hlayout)
        self.back.setMinimumHeight(40)
        self.next.setMinimumHeight(40)
        self.randbtn.setMinimumHeight(40)
        self.index_editer.setMinimumHeight(40)
        self.index_editer.setMaximumWidth(80)
        self.index_editer.setMaxLength(8)

    def set_dataset_info(self):
        if self.dataSet is None:
            self.datasetInfo.setText(f"Please open.")
        elif len(self.dataSet) == 0:
            self.datasetInfo.setText(f"No image in folder.")
        else:
            self.datasetInfo.setText(f"Images Index From 0 to {len(self.dataSet) - 1}.")

    def set_image_info(self, p='None'):
        self.imgInfo.setText(f'Image Info: {p}')

    def getImage(self):
        self.img, c = self.dataSet[self.index]
        p, _ = self.dataSet.samples[self.index]
        self.set_image_info(os.path.basename(p))
        self.img = toTensor(self.img).unsqueeze(0)
        if self.send is not None:
            self.send(self.img)

    def set_index(self, i):
        i = i % len(self.dataSet)
        self.index = i
        self.index_editer.setText(str(i))
        self.getImage()

    def parse_index(self):
        t = self.index_editer.text()
        try:
            i = int(t)
            self.set_index(i)
        except Exception() as e:
            pass

    def indexNext(self):
        self.set_index(self.index + 1)

    def indexBack(self):
        self.set_index(self.index - 1)

    def indexRand(self):
        self.set_index(random.randint(0, len(self.dataSet) - 1) + 1)

    def set_dateset(self, ds):
        self.dataSet = ds
        self.set_dataset_info()
        self.set_index(0)

    def link(self, sendCallBack):
        self.send = None
        if sendCallBack is not None:
            self.send = sendCallBack
