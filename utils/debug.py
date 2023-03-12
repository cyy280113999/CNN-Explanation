import torch
import matplotlib.pyplot as plt

from .image_process import invStd
from .plot import toPlot, lrp_cmap

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


# show tensor
def showTensorImgPositive(tensor):
    fig = plt.figure()
    plt.imshow(toPlot(tensor.cpu()), vmin=0., vmax=1., cmap=lrp_cmap)
    plt.axis(False)
    plt.show()

def showTensorImgReal(tensor):
    fig = plt.figure()
    plt.imshow(toPlot(tensor.cpu()), vmin=-1., vmax=1., cmap=lrp_cmap)
    plt.axis(False)
    plt.show()


def showVRAM():
    print(torch.cuda.memory_summary())
    torch.cuda.empty_cache()
