import numpy as np
import torch
import torch.nn.functional as nf
import torchvision
from PIL import Image
import matplotlib.colors
import matplotlib.pyplot as plt
import pyqtgraph as pg

device = 'cuda'


# ========== pil image loading
def pilOpen(filename):
    return Image.open(filename).convert('RGB')


# =========== image process
toRRC = torchvision.transforms.RandomResizedCrop(224, scale=(0.25, 1), ratio=(1, 1))

toTensorS224 = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor()
])

toTensor = torchvision.transforms.ToTensor()

ImgntMean = [0.485, 0.456, 0.406]
ImgntStd = [0.229, 0.224, 0.225]
ImgntMeanTensor = torch.tensor(ImgntMean).reshape(1, -1, 1, 1)
ImgntStdTensor = torch.tensor(ImgntStd).reshape(1, -1, 1, 1)

toStd = torchvision.transforms.Normalize(ImgntMean, ImgntStd)


def invStd(tensor):
    tensor = tensor * ImgntStdTensor + ImgntMeanTensor
    return tensor


# ============ std image loading
# image for raw (PIL,numpy) image. x for standardized tensor
def get_image_x(filename='cat_dog_243_282.png', image_folder='input_images/', device=device):
    # require folder , pure filename
    filename = image_folder + filename
    img_PIL = pilOpen(filename)
    img_tensor = toTensorS224(img_PIL).unsqueeze(0)
    img_tensor = toStd(img_tensor).to(device)
    return img_tensor


# =========== imagenet loading
imageNetDefaultDir = r'F:/DataSet/imagenet/'
imageNetSplits = {
    'train': 'train/',
    'val': 'val/',
}

default_transform = torchvision.transforms.Compose([
    toTensorS224,
    toStd
])


def loadImageNetClasses(path=imageNetDefaultDir):
    import json
    filename = path + 'imagenet_class_index.json'
    with open(filename) as f:
        c = json.load(f)
        c = {int(i): v[-1] for i, v in c.items()}
        return c


def getImageNet(split, transform=default_transform):
    return torchvision.datasets.ImageNet(root=imageNetDefaultDir,
                                         split=split,
                                         transform=transform)


# ======== plot functions
def toPlot(x):
    # 'toPlot' is to inverse the operation of 'toTensor'
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


# ============== heatmap process
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
            raise Exception(f'no layer:({layer_name}) in model')
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
    if not hasattr(obj, 'hooks'):
        obj.hooks = []
    if layer_name == 'input_layer':
        obj.hooks.append(model.register_forward_hook(lambda *args: save_act_in(obj, *args)))
        obj.hooks.append(model.register_full_backward_hook(lambda *args: save_grad_in(obj, *args)))
    else:
        layer = findLayerByName(model, layer_name)
        obj.hooks.append(layer.register_forward_hook(lambda *args: forward_hook(obj, *args)))
        obj.hooks.append(layer.register_full_backward_hook(lambda *args: backward_hook(obj, *args)))


def relevanceFindByName(model, layer_name=(None,)):
    # compatible for input layer
    if layer_name=='input_layer' or layer_name[0]=='input_layer':
        return model.x.diff(dim=0) * model.g
    else:
        layer = findLayerByName(model, layer_name)
        return layer.y.diff(dim=0) * layer.g


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


def interpolate_to_imgsize(heatmap):  # only for heatmap
    return heatmapNormalizeR(nf.interpolate(heatmap.sum(1, True), 224, mode='bilinear'))


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


# ========== window
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


