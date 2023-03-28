import torch
from .plot import toPlot, lrp_cmap,lrp_lut, plotItemDefaultConfig,pyqtgraphDefaultConfig
import pyqtgraph as pg
import matplotlib.pyplot as plt

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
def showHeatmap(x):
    pyqtgraphDefaultConfig()
    glw=pg.GraphicsLayoutWidget()
    pi=glw.addPlot()
    x=x/x.abs().max()
    x=toPlot(x.sum(1,True))
    ii=pg.ImageItem(x,levels=[-1, 1], lut=lrp_lut)
    pi.addItem(ii)
    plotItemDefaultConfig(pi)
    glw.show()
    pg.exec()


def showImg(tensor):
    glw = pg.GraphicsLayoutWidget()
    pim = glw.addPlot()
    ii = pg.ImageItem(toPlot(tensor))
    pim.addItem(ii)
    glw.show()
    pg.exec()


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
