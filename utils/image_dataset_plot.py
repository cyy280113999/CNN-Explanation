import numpy as np
import torch
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
            x = x.squeeze(0)
        # case `(H, W)`
        if len(x.shape) == 2:
            x.reshape((1,)+x.shape)
        if len(x.shape) != 3:
            raise TypeError('mismatch dimension')
        # case `(C, H, W)`
        return x.transpose(1, 2, 0) # hwc
    else:
        raise TypeError(f'Plot Type is unavailable for {type(x)}')


# image data process
def heatmapNormalizeP(tensor):
    low = tensor.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    hig = tensor.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    hig += ((hig-low)<=1e-9)*1e-9
    tensor = (tensor - low) / (hig - low)
    return tensor


def heatmapNormalizeR(heatmap):
    return heatmap / heatmap.abs().max()


def heatmapNormalizeR2P(heatmap):
    peak = heatmap.abs().max()
    # heatmap = ((heatmap/peak)+1)/2
    return heatmap / peak / 2 + 0.5


