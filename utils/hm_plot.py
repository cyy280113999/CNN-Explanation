import matplotlib.colors
import matplotlib.pyplot as plt
import pyqtgraph as pg
import torch
import torch.nn.functional as nf
import numpy as np


# ============== heatmap process
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
