import numpy as np
import torch
import torch.nn.functional as nf
import torchvision
from PIL import Image

device = 'cuda'
imageNetDefaultDir = r'F:/DataSet/imagenet/'
imageNetSplits = {
    'train': 'train/',
    'val': 'val/',
}


def pilOpen(filename):
    return Image.open(filename).convert('RGB')


toRRC = torchvision.transforms.RandomResizedCrop(224, scale=(0.25, 1), ratio=(1, 1))

toTensorS224 = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor()
])

ImgntMean = [0.485, 0.456, 0.406]
ImgntStd = [0.229, 0.224, 0.225]
ImgntMeanTensor = torch.tensor(ImgntMean).reshape(1, -1, 1, 1)
ImgntStdTensor = torch.tensor(ImgntStd).reshape(1, -1, 1, 1)

toStd = torchvision.transforms.Normalize(ImgntMean, ImgntStd)


def invStd(tensor):
    tensor = tensor * ImgntStdTensor + ImgntMeanTensor
    return tensor


# image for raw (PIL,numpy) image. x for standardized tensor
def get_image_x(filename='cat_dog_243_282.png', image_folder='input_images/', device=device):
    # require folder , pure filename
    filename = image_folder + filename
    img_PIL = pilOpen(filename)
    img_tensor = toTensorS224(img_PIL).unsqueeze(0)
    img_tensor = toStd(img_tensor).to(device)
    return img_tensor


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
def heatmapNormalizeR(heatmap):
    M = heatmap.abs().max()
    heatmap = heatmap / M
    heatmap = torch.nan_to_num(heatmap)
    return heatmap


def heatmapNormalizeR_every(heatmap):
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
