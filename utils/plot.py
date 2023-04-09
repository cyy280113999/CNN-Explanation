import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import pyqtgraph as pg
from .image_dataset_plot import invStd, toPlot
from .masking import binarize

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


pyqtgraphDefaultConfig()


def plotItemDefaultConfig(p):
    # this for showing image
    p.showAxes(False)
    p.invertY(True)
    p.vb.setAspectLocked(True)


# def save_fig(save_path, subplots):
#     num_subplots = len(subplots)
#     fig = plt.figure(figsize=(3 * num_subplots, 3))
#     for i, (title, images) in enumerate(subplots):
#         ax = fig.add_subplot(1, num_subplots, i + 1)
#         ax.set_axis_off()
#         for image, cmap, alpha in images:
#             ax.imshow(image, cmap=cmap, alpha=alpha, vmin=0., vmax=1.)
#         ax.set_title(title)
#     plt.tight_layout()
#     if save_path is not None:
#         plt.savefig(save_path)
#     plt.close(fig)


# def save_plot(std_img=None, heatmap=None, save_path=None, cmap='lrp', alpha=0.5):
#     """ Method to plot the explanation.
#         input_: ImgntStdTensor image.
#         heatmap:p-n heatmap
#         save_path: String. Defaults to None.
#         cmap: 'jet'. p-red , n-blue , zero-green
#         alpha: Defaults to be 0.5.
#
#         cam_map = cam(input_, class_idx=class_idx, sg=sg, norm=norm, relu=relu)
#         save_path = save_root + f'{image_name}' \
#                                 f'{("_cl" + str(class_idx)) if class_idx else ""}' \
#                                 f'{"_norm" if norm else ""}{"_sg" if sg else ""}' \
#                                 f'{"_relu"if relu else""}.png'
#         visualize(input_.cpu().detach(), cam_map.type(torch.FloatTensor).cpu().detach(),
#                   save_path=save_path)
#     """
#     if cmap == 'lrp':
#         cmap = lrp_cmap
#     subplots = []
#     if std_img is not None:
#         std_img = toPlot(invStd(std_img))
#         subplots.append(('Input Image', [(std_img, None, None)]))
#     if heatmap is not None:
#         heatmap = toPlot(heatmapNormalizeR2P(heatmap))
#         subplots.append(('Heat Map', [(heatmap, cmap, None)]))
#     if std_img is not None and heatmap is not None:
#         subplots.append(('Overlay', [(std_img, None, None), (heatmap, cmap, alpha)]))
#
#     save_fig(save_path, subplots)


# def save_masking(std_img, heatmap, save_path=None, cmap='rainbow', sparsity=0.5):
#     subplots = []
#
#     std_img = invStd(std_img)
#     subplots.append(('Input image', [(toPlot(std_img), None, None)]))
#     subplots.append(('Mask', [(toPlot(heatmapNormalizeR2P(heatmap)), cmap, None)]))
#     subplots.append(('Masking', [(toPlot(std_img * binarize(heatmap, sparsity=sparsity)), None, None)]))
#
#     save_fig(save_path, subplots)
#
#
# def save_softmasking(std_img, heatmap, save_path=None, cmap='rainbow', alpha=0.5):
#     subplots = []
#
#     std_img = invStd(std_img)
#     subplots.append(('Input image', [(toPlot(std_img), None, None)]))
#     subplots.append(('Mask', [(toPlot(heatmapNormalizeR2P(heatmap)), cmap, None)]))
#     subplots.append(('Soft Masking', [(toPlot(std_img * heatmapNormalizeR(heatmap)), None, None)]))
#
#     save_fig(save_path, subplots)

