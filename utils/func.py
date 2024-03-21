import os
import time
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import pyqtgraph as pg

from .hm_plot import toPlot, heatmapNormalizeR, lrp_lut, lrp_cmap_gl, plotItemDefaultConfig


# make path (recursively)
def mkp(p):
    d = os.path.dirname(p)
    if d not in ['', '.', '..'] and not os.path.exists(d):
        mkp(d)
        os.mkdir(d)


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
    torch.save(tensor, path)


def save_image(tensor, path):
    ii = pg.ImageItem(toPlot(tensor), levels=[0, 1])
    ii.save(path)


def save_heatmap(tensor, path):
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
