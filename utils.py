import os

import torchvision
import numpy as np
import torch
from PIL import Image
import matplotlib.colors
import matplotlib.pyplot as plt
import itertools
import tqdm
from torchvision.models import VGG16_Weights
from time import time

device = 'cuda'


def get_model(device=device):
    return torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval().to(device)


transToTensor = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor()
])

RRCTensor = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224, scale=(0.25, 1), ratio=(1, 1)),
    torchvision.transforms.ToTensor()
])

ImgntMean = [0.485, 0.456, 0.406]
ImgntStd = [0.229, 0.224, 0.225]
transToStd = torchvision.transforms.Normalize(ImgntMean, ImgntStd)


def get_image(filename='cat_dog_243_282.png', image_folder='input_images/', device=device):
    # require folder , pure filename
    filename = image_folder + filename
    img_PIL = Image.open(filename).convert('RGB')
    img_tensor = transToTensor(img_PIL).unsqueeze(0)
    return transToStd(img_tensor).to(device)


def tensorInfo(tensor, print_info=True):
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


def InverseStd(tensor):
    mean, std = torch.tensor(ImgntMean).reshape(1, -1, 1, 1), torch.tensor(ImgntStd).reshape(1, -1, 1, 1)
    tensor = tensor * std + mean
    return tensor


def ToPlot(tensor):
    if isinstance(tensor, torch.Tensor):
        # case `(N, C, H, W)`
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        # case `(H, W)`
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)
        # case `(C, H, W)`
        if len(tensor.shape) == 3:
            return tensor.permute(1, 2, 0)
        else:
            raise TypeError('mismatch dimension')
    else:
        raise TypeError(f'Plot Type is unavailable for {type(tensor)}')


# image data process

def normalize(tensor):
    low = tensor.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    hig = tensor.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    hig += 1e-9
    assert (hig != low).all()
    tensor = (tensor - low) / (hig - low)
    return tensor


def normalize_R(heatmap):
    return heatmap / heatmap.abs().max()


def normalize_R2P(heatmap):
    peak = heatmap.abs().max()
    # heatmap = ((heatmap/peak)+1)/2
    return heatmap / peak / 2 + 0.5


def binarize(mask, sparsity=0.5):
    x = mask.flatten()
    n = len(x)
    ascending = x.sort()[0]
    value = ascending[int(sparsity * n)]
    return 1.0 * (mask >= value)


def positize(mask):
    return 1.0 * (mask >= 0)


def binarize_noisy(mask, sparsity=0.5):
    x = mask.flatten()
    n = len(x)
    ascending = x.sort()[0]
    value = ascending[int(sparsity * n)]
    return 1.0 * (mask >= value) + (2 * torch.rand_like(mask) - 1) * (mask < value)


def maximalPatch(mask2d, top=True, r=1):
    if len(mask2d.shape) == 4:
        mask2d = mask2d.squeeze(0)
    if len(mask2d.shape) == 3:
        mask2d = mask2d.sum(0)
    assert len(mask2d.shape) == 2
    h, w = mask2d.shape
    f = mask2d.flatten()
    loc = f.sort(descending=top)[1][0].item()
    x = loc // h
    y = loc - (x * h)
    xL = max(0, x - r)
    xH = min(h, x + r)
    yL = max(0, y - r)
    yH = min(w, y + r)
    patched = torch.ones_like(mask2d)
    patched[xL:xH + 1, yL:yH + 1] = 0
    return patched


# image save

lrp_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
lrp_cmap[:, 0:3] *= 0.85
lrp_cmap = matplotlib.colors.ListedColormap(lrp_cmap)


def visualize(std_img=None, heatmap=None, save_path=None, cmap='lrp', alpha=0.5):
    """ Method to plot the explanation.
        input_: std image.
        heatmap:p-n heatmap
        save_path: String. Defaults to None.
        cmap: 'jet'. p-red , n-blue , zero-green
        alpha: Defaults to be 0.5.

        cam_map = cam(input_, class_idx=class_idx, sg=sg, norm=norm, relu=relu)
        save_path = save_root + f'{image_name}' \
                                f'{("_cl" + str(class_idx)) if class_idx else ""}' \
                                f'{"_norm" if norm else ""}{"_sg" if sg else ""}' \
                                f'{"_relu"if relu else""}.png'
        visualize(input_.cpu().detach(), cam_map.type(torch.FloatTensor).cpu().detach(),
                  save_path=save_path)
    """
    if cmap == 'lrp':
        cmap = lrp_cmap
    subplots = []
    if std_img is not None:
        std_img = ToPlot(InverseStd(std_img))
        subplots.append(('Input image', [(std_img, None, None)]))
    if heatmap is not None:
        heatmap = ToPlot(normalize_R2P(heatmap))
        subplots.append(('Saliency map across RGB channels', [(heatmap, cmap, None)]))
    if std_img is not None and heatmap is not None:
        subplots.append(('Overlay', [(std_img, None, None), (heatmap, cmap, alpha)]))

    save_fig(save_path, subplots)


def save_fig(save_path, subplots):
    num_subplots = len(subplots)
    fig = plt.figure(figsize=(3 * num_subplots, 3))
    for i, (title, images) in enumerate(subplots):
        ax = fig.add_subplot(1, num_subplots, i + 1)
        ax.set_axis_off()
        for image, cmap, alpha in images:
            ax.imshow(image, cmap=cmap, alpha=alpha, vmin=0., vmax=1.)
        ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)


def visualize_masking(std_img, heatmap, save_path=None, cmap='rainbow', sparsity=0.5):
    subplots = []

    std_img = InverseStd(std_img)
    subplots.append(('Input image', [(ToPlot(std_img), None, None)]))
    subplots.append(('Mask', [(ToPlot(normalize_R2P(heatmap)), cmap, None)]))
    subplots.append(('Masking', [(ToPlot(std_img * binarize(heatmap, sparsity=sparsity)), None, None)]))

    save_fig(save_path, subplots)


def visualize_softmasking(std_img, heatmap, save_path=None, cmap='rainbow', alpha=0.5):
    subplots = []

    std_img = InverseStd(std_img)
    subplots.append(('Input image', [(ToPlot(std_img), None, None)]))
    subplots.append(('Mask', [(ToPlot(normalize_R2P(heatmap)), cmap, None)]))
    subplots.append(('Soft Masking', [(ToPlot(std_img * normalize(heatmap)), None, None)]))

    save_fig(save_path, subplots)


# show tensor
def show(img):
    fig = plt.figure()
    plt.imshow(ToPlot(InverseStd(img.cpu())), vmin=0., vmax=1.)
    plt.axis(False)
    plt.show()


# make path (recursively)
def mkp(p):
    d = os.path.dirname(p)
    if d not in ['', '.', '..'] and not os.path.exists(d):
        mkp(d)
        os.mkdir(d)


# running cost
class RunningCost:
    def __init__(self, stage_count=5):
        self.stage_count = stage_count
        self.running_cost = [None for i in enumerate(range(self.stage_count + 1))]
        self.hint = [None for i in enumerate(range(self.stage_count))]
        self.position = 0

    def tic(self, hint=None):
        if self.position < self.stage_count:
            t = time()
            self.running_cost[self.position] = t
            self.hint[self.position]=hint
            self.position += 1

    def cost(self):
        print('-'*20)
        for stage_, (i, j) in enumerate(zip(self.running_cost, self.running_cost[1:])):
            if j is not None:
                if self.hint[stage_+1] is not None:
                    print(f'stage {self.hint[stage_+1]} cost time: {j - i}')
                else:
                    print(f'stage {stage_+1} cost time: {j - i}')

        print('-' * 20)