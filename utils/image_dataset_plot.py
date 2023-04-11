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


# heatmap process
def FindLayerByName(model, layer_name=(None,)):
    layer = model
    if not isinstance(layer_name, (tuple, list)):
        layer_name = (layer_name,)
    for l in layer_name:
        if isinstance(l, int):  # for sequential
            layer = layer[l]
        elif isinstance(l, str) and hasattr(layer, l):  # for named child
            layer = layer.__getattr__(l)
        else:
            raise Exception()
    return layer


def RelevanceFindByName(model, layer_name=(None,)):
    layer = FindLayerByName(model,layer_name)
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


def interpolate_to_imgsize(heatmap): # only for heatmap
    return heatmapNormalizeR(nf.interpolate(heatmap.sum(1, True), 224, mode='bilinear'))


def multi_interpolate(heatmaps):
    return heatmapNormalizeR(sum(interpolate_to_imgsize(x) for x in heatmaps))


# ============ drawing
# image save
lrp_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
lrp_cmap[:, 0:3] *= 0.85
lrp_cmap = matplotlib.colors.ListedColormap(lrp_cmap)

lrp_cmap_gl=pg.colormap.get('seismic',source='matplotlib')
# lrp_cmap_gl.color[:,0:3]*=0.8
# lrp_cmap_gl.color[2,3]=0.5
lrp_lut=lrp_cmap_gl.getLookupTable(start=0,stop=1,nPts=256)


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


# ============= debug
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
    glw=pg.GraphicsLayoutWidget()
    pi=glw.addPlot()
    x=toPlot(heatmapNormalizeR(x.sum(1,True)))
    ii=pg.ImageItem(x,levels=[-1, 1], lut=lrp_lut)
    pi.addItem(ii)
    plotItemDefaultConfig(pi)
    glw.show()
    pg.exec()


def show_image(tensor):
    glw = pg.GraphicsLayoutWidget()
    pim = glw.addPlot()
    ii = pg.ImageItem(toPlot(tensor))
    pim.addItem(ii)
    glw.show()
    pg.exec()


def save_tensor(tensor, path):
    torch.save(tensor, path)


def save_image(tensor, path):
    ii = pg.ImageItem(toPlot(tensor))
    ii.save(path)


def save_heatmap(tensor, path):
    tensor = toPlot(heatmapNormalizeR(tensor.sum(1, True)))
    ii = pg.ImageItem(tensor, levels=[-1, 1], lut=lrp_lut)
    ii.save(path)


def showVRAM():
    print(torch.cuda.memory_summary())
    torch.cuda.empty_cache()

def histogram(x):
    if isinstance(x,torch.Tensor):
        x = x.detach()
        if x.device.type=='cuda':
            x = x.cpu()
    # histogram
    a,b = x.histogram()
    a = a.numpy()
    width = b[1]-b[0]
    b += width
    b = b[:-1].numpy()
    plt.bar(b,a,width)
    plt.show()
