import numpy as np
import torch
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import pyqtgraph as pg
from .image_process import invStd
from .masking import binarize

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


# image save
lrp_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
lrp_cmap[:, 0:3] *= 0.85
lrp_cmap = matplotlib.colors.ListedColormap(lrp_cmap)

lrp_cmap_gl=pg.colormap.get('seismic',source='matplotlib')
# lrp_cmap_gl.color[:,0:3]*=0.8
# lrp_cmap_gl.color[2,3]=0.5
lrp_lut=lrp_cmap_gl.getLookupTable(start=0,stop=1,nPts=256)


def pyqtgraphDefaultConfig():
    pg.setConfigOptions(**{'imageAxisOrder': 'row-major',
                           'background': 'w',
                           'foreground': 'k',
                           # 'useNumba': True,
                           # 'useCupy': True,
                           })

def plotItemDefaultConfig(p):
    # this for showing image
    p.showAxes(False)
    p.invertY(True)
    p.vb.setAspectLocked(True)


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


def visualize(std_img=None, heatmap=None, save_path=None, cmap='lrp', alpha=0.5):
    """ Method to plot the explanation.
        input_: ImgntStdTensor image.
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
        std_img = toPlot(invStd(std_img))
        subplots.append(('Input Image', [(std_img, None, None)]))
    if heatmap is not None:
        heatmap = toPlot(heatmapNormalizeR2P(heatmap))
        subplots.append(('Heat Map', [(heatmap, cmap, None)]))
    if std_img is not None and heatmap is not None:
        subplots.append(('Overlay', [(std_img, None, None), (heatmap, cmap, alpha)]))

    save_fig(save_path, subplots)


def visualize_masking(std_img, heatmap, save_path=None, cmap='rainbow', sparsity=0.5):
    subplots = []

    std_img = invStd(std_img)
    subplots.append(('Input image', [(toPlot(std_img), None, None)]))
    subplots.append(('Mask', [(toPlot(heatmapNormalizeR2P(heatmap)), cmap, None)]))
    subplots.append(('Masking', [(toPlot(std_img * binarize(heatmap, sparsity=sparsity)), None, None)]))

    save_fig(save_path, subplots)


def visualize_softmasking(std_img, heatmap, save_path=None, cmap='rainbow', alpha=0.5):
    subplots = []

    std_img = invStd(std_img)
    subplots.append(('Input image', [(toPlot(std_img), None, None)]))
    subplots.append(('Mask', [(toPlot(heatmapNormalizeR2P(heatmap)), cmap, None)]))
    subplots.append(('Soft Masking', [(toPlot(std_img * heatmapNormalizeP(heatmap)), None, None)]))

    save_fig(save_path, subplots)

