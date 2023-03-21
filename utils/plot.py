import numpy as np
import torch
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import pyqtgraph as pg
from .image_process import invStd
from .masking import binarize

def toPlot(x):
    if isinstance(x, torch.Tensor):
        if torch.get_device(x) == torch.device('cuda'):
            x = x.detach().cpu()
        # case `(N, C, H, W)`
        if len(x.shape) == 4:
            x = x.squeeze(0)
        # case `(H, W)`
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        # case `(C, H, W)`
        if len(x.shape) == 3:
            return x.permute(1, 2, 0)
        else:
            raise TypeError('mismatch dimension')
    elif isinstance(x, np.ndarray):
        if len(x.shape) == 4:
            x = x.squeeze(0)
        if len(x.shape) == 2:
            x.reshape((1,)+x.shape)
        if len(x.shape) == 3:
            return x.transpose(1, 2, 0)
        else:
            raise Exception()
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


# def convertFromMatplotlib(col_map,name=''):
#     import numpy as np
#     from collections.abc import Callable, Sequence
#     from pyqtgraph.colormap import ColorMap
#     import matplotlib.pyplot as mpl_plt
#     cmap = None
#     if hasattr(col_map, '_segmentdata'): # handle LinearSegmentedColormap
#         data = col_map._segmentdata
#         if ('red' in data) and isinstance(data['red'], (Sequence, np.ndarray)):
#             positions = set() # super-set of handle positions in individual channels
#             for key in ['red','green','blue']:
#                 for tup in data[key]:
#                     positions.add(tup[0])
#             col_data = np.zeros((len(positions),4 ))
#             col_data[:,-1] = sorted(positions)
#             for idx, key in enumerate(['red','green','blue']):
#                 positions = np.zeros( len(data[key] ) )
#                 comp_vals = np.zeros( len(data[key] ) )
#                 for idx2, tup in enumerate( data[key] ):
#                     positions[idx2] = tup[0]
#                     comp_vals[idx2] = tup[1] # these are sorted in the raw data
#                 col_data[:,idx] = np.interp(col_data[:,3], positions, comp_vals)
#             cmap = ColorMap(pos=col_data[:,-1], color=255*col_data[:,:3]+0.5)
#         # some color maps (gnuplot in particular) are defined by RGB component functions:
#         elif ('red' in data) and isinstance(data['red'], Callable):
#             col_data = np.zeros((64, 4))
#             col_data[:,-1] = np.linspace(0., 1., 64)
#             for idx, key in enumerate(['red','green','blue']):
#                 col_data[:,idx] = np.clip( data[key](col_data[:,-1]), 0, 1)
#             cmap = ColorMap(pos=col_data[:,-1], color=255*col_data[:,:3]+0.5)
#     elif hasattr(col_map, 'colors'): # handle ListedColormap
#         col_data = np.array(col_map.colors)
#         cmap = ColorMap(name=name, pos=np.linspace(0.0, 1.0, col_data.shape[0]), color=255*col_data[:,:3]+0.5 )
#     if cmap is not None:
#         cmap.name = name
#     return cmap
# lrp_cmap_gl = convertFromMatplotlib(lrp_cmap,'lrp_cmap')
# lrp_lut=lrp_cmap_gl.getLookupTable(start=0,stop=1,nPts=256)
# colors = [#all black
#     (0, 0, 1),  # 蓝色
#     (1, 1, 1),  # 白色
#     (1, 0, 0)   # 红色
# ]
# pos = np.linspace(-1, 1, 256)
# lrp_cmap_gl = pg.ColorMap(name='lrp_cmap', pos=[-1, 0, 1], color=colors)
# lrp_lut=lrp_cmap_gl.getLookupTable(start=0,stop=1,nPts=256)
# pos = np.linspace(0, 1, 256)
# colors = np.array([(0, 0, 1), (1, 1, 1), (1, 0, 0)])
# colors = np.interp(pos, , )
# lrp_cmap_gl = pg.ColorMap(pos=pos, color=colors)
# lrp_lut = lrp_cmap_gl.getLookupTable(start=0, stop=1, nPts=256)
lrp_cmap_gl=pg.colormap.get('seismic',source='matplotlib')
# lrp_cmap_gl.color[:,0:3]*=0.8
# lrp_cmap_gl.color[2,3]=0.5
lrp_lut=lrp_cmap_gl.getLookupTable(start=0,stop=1,nPts=256)


def pyqtgraphDefaultConfig():
    pg.setConfigOptions(**{'imageAxisOrder': 'row-major',
                           'background': 'w',
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

