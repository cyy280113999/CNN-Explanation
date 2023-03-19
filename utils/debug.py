import torch
import pyqtgraph as pg
from .plot import toPlot, lrp_cmap, pyqtgraphDefaultConfig, plotItemDefaultConfig
pyqtgraphDefaultConfig(pg)

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
def showTensorImg(tensor):
    glw=pg.GraphicsLayoutWidget()
    pi=glw.addPlot()
    ii=pg.ImageItem(toPlot(tensor.cpu().detach().numpy()))
    pi.addItem(ii)
    plotItemDefaultConfig(pi)
    glw.show()
    pg.exec()


def showTensorImgReal(tensor):
    glw = pg.GraphicsLayoutWidget()
    pim = glw.addPlot()
    ii = pg.ImageItem(toPlot(tensor.cpu()), levels=[-1, 1])
    pim.addItem(ii)
    glw.show()
    pg.exec()


def showVRAM():
    print(torch.cuda.memory_summary())
    torch.cuda.empty_cache()
