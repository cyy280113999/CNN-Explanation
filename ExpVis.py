# sys
import os
import sys
import random
from functools import partial
from math import ceil
# nn
import numpy as np
import torch as tc
import torch.cuda
import torch.nn.functional as nf
import torchvision as tv
# gui
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QPainter
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QGroupBox, QVBoxLayout, QComboBox, QPushButton, QLineEdit, \
    QFileDialog, QMainWindow, QApplication
# draw
from PIL import Image

USING_DRAW_BACKEND = 'gl'
if USING_DRAW_BACKEND == 'mpl':
    import matplotlib as mpl
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    # mpl initialize
    mpl.use('QtAgg')  # 指定渲染后端。QtAgg后端指用Agg二维图形库在Qt控件上绘图。
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    # mpl.rcParams['figure.dpi'] = 400
elif USING_DRAW_BACKEND == 'gl':
    import pyqtgraph as pg
    # import pyqtgraph.opengl as gl
    import pyqtgraph.colormap as pcolors

    pg.setConfigOptions(**{'imageAxisOrder': 'row-major',
                           # 'useNumba': True,
                           # 'useCupy': True,
                           })

    from pyqtgraph.widgets.RawImageWidget import RawImageWidget
    from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget

# user
from utils import *
from datasets.OnlyImages import OnlyImages
from datasets.DiscrimDataset import *
from methods.cam.gradcam import GradCAM
from methods.cam.layercam import LayerCAM
from methods.RelevanceCAM import RelevanceCAM
from methods.scorecam import ScoreCAM
from methods.AblationCAM import AblationCAM
from methods.Taylor import Taylor
from methods.LRP import LRP_Generator
from methods.LRP_0 import LRP_0
from methods.LIDLinearDecompose import LIDLinearDecomposer
from methods.LIDIGDecompose import LIDIGDecomposer
from methods.IG import IGDecomposer

if USING_DRAW_BACKEND == 'gl':
    # import matplotlib as mpl
    # from matplotlib.figure import Figure
    # import matplotlib.pyplot as plt
    # lrp_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    # lrp_cmap[:, 0:3] *= 0.85
    # cmap = pcolors.ColorMap(name='lrp_cmap',pos=np.linspace(0.0, 1.0, col_data.shape[0]), color=255 * col_data[:, :3] + 0.5)
    # lrp_cmap = matplotlib.colors.ListedColormap(lrp_cmap)
    # lrp_cmap_gl = pcolors.get('bwr', 'matplotlib')
    pass

# torch initial
device = "cuda"

if USING_DRAW_BACKEND == 'mpl':
    # figureWidget=FigureCanvasQTAgg(figure)
    # testing..
    def loadTestImg(filename="testImg.png"):
        img_PIL = Image.open(filename).convert('RGB')
        img = np.asarray(img_PIL)
        return img


    def loadTestImgFigureAxe(filename="testImg.png"):
        img = loadTestImg(filename)
        figure = Figure()
        axe = figure.add_subplot()  # create a axe in figure
        axe.imshow(img)
        return figure, axe


    def loadTestImgWidget(filename="testImg.png"):
        img = loadTestImg(filename)
        figure = plt.figure()
        plt.imshow(img)
        # 容器层包含（1）画板层Canvas（2）画布层Figure（3）绘图区 / 坐标系Axes
        figureWidget = FigureCanvasQTAgg(figure)
        return figureWidget


class TipedWidget(QWidget):
    def __init__(self, tip="Empty Tip", widget=None):
        super(TipedWidget, self).__init__()
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        main_layout.addWidget(QLabel(tip))
        if widget is None:
            raise Exception("Must given widget.")
        self.widget = widget
        main_layout.addWidget(self.widget)

    def __getitem__(self, item):
        return self.widget.__getitem(item)


class ImageCanvas(QGroupBox):
    def __init__(self):
        super(ImageCanvas, self).__init__()
        self.setTitle("Image:")
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        if USING_DRAW_BACKEND == 'mpl':
            self.figure = Figure()
            self.axe = None
            self.canvas = FigureCanvasQTAgg(self.figure)
            main_layout.addWidget(self.canvas)
        elif USING_DRAW_BACKEND == 'gl':
            # self.imv = pg.ImageView()
            # self.imv.setImage()
            self.pglw = pg.GraphicsLayoutWidget()
            main_layout.addWidget(self.pglw)

    def showImage(self, img):
        if USING_DRAW_BACKEND == 'mpl':
            self.figure.clf()
            self.axe = self.figure.add_subplot()
            self.axe.imshow(img)
            self.axe.set_axis_off()
            self.figure.tight_layout()
            self.canvas.draw_idle()  # 刷新
        elif USING_DRAW_BACKEND == 'gl':
            if isinstance(img, Image.Image):
                img = np.asarray(img)
            elif isinstance(img, tc.Tensor):
                img = img.numpy()
            imi = pg.ImageItem(img, autolevel=False, autorange=False)  # ,levels=[0,1])#,lut=None)
            # cmap=pcolors.get()
            self.pglw.clear()
            p: pg.PlotItem = self.pglw.addPlot()
            p.addItem(imi)
            plotItemDefaultConfig(p)

    if USING_DRAW_BACKEND == 'mpl':
        def showFigure(self, fig):
            self.axe = None
            if self.figure is not fig:
                self.figure = fig
            self.canvas.figure = self.figure
            self.canvas.draw_idle()  # 刷新

    def showImages(self, imgs):
        pass

    def clear(self):
        if USING_DRAW_BACKEND == 'mpl':
            self.figure = None
            self.canvas.figure = None
        elif USING_DRAW_BACKEND == 'gl':
            self.imv.clear()

    # ValueError: The Axes must have been created in the present figure
    # def showAxe(self, axe):
    #     self.figure.clf()
    #     self.axe=axe
    #     self.figure.add_axes(axe)
    #     # if self.axe is None:
    #     #     self.axe = self.figure.add_subplot()
    #     #     # axe = self.figure.add_axes() # Error
    #     # self.axe.clear()
    #     # # self.axe.imshow(img)
    #     self.axe.set_axis_off()
    #     self.canvas.draw_idle()


class ImageLoader(QGroupBox):
    def __init__(self):
        super().__init__()
        # key: dataset name , value: is folder or not
        self.imageNetVal = tv.datasets.ImageNet(root="F:/DataSet/imagenet", split="val")
        self.classes = loadImageNetClasses()
        self.dataSets = {
            "Customized Image": None,
            "Customized Folder": None,
            "ImageNet Val": lambda: self.imageNetVal,
            "ImageNet Train": partial(tv.datasets.ImageNet, root="F:/DataSet/imagenet", split="train"),
            "Discrim DataSet": lambda: DiscrimDataset(discrim_hoom, discrim_imglist, False),
        }
        self.dataSet = None
        self.img = None
        # self.setMaximumWidth(600)
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        self.setTitle("Image Loader")

        # data set
        self.dataSetSelect = QComboBox()
        # self.dataSetSelect.resize(200,40)
        self.dataSetSelect.setMinimumHeight(40)
        # self.dataSetSelect
        items = QStandardItemModel()
        for key in self.dataSets:
            item = QStandardItem(key)
            item.setData(key)  # , Qt.ToolTipRole
            item.setSizeHint(QSize(200, 40))
            items.appendRow(item)
        self.dataSetSelect.setModel(items)
        self.dataSetSelect.setCurrentIndex(0)

        main_layout.addWidget(TipedWidget("Data Set: ", self.dataSetSelect))

        # data set info
        self.dataSetLen = QLabel("")
        main_layout.addWidget(self.dataSetLen)

        # image choose
        hlayout = QHBoxLayout()
        self.open = QPushButton("Open")  # no folder only
        self.back = QPushButton("Back")
        self.next = QPushButton("Next")
        self.index = QLineEdit("0")
        hlayout.addWidget(self.open)
        hlayout.addWidget(self.back)
        hlayout.addWidget(self.next)
        hlayout.addWidget(self.index)
        main_layout.addLayout(hlayout)
        del hlayout

        # self.open.setFixedSize(80,40)
        # self.back.setFixedSize(80,40)
        # self.next.setFixedSize(80,40)
        self.open.setMinimumHeight(40)
        self.back.setMinimumHeight(40)
        self.next.setMinimumHeight(40)
        self.index.setMinimumHeight(40)
        self.index.setMaximumWidth(80)
        self.index.setMaxLength(8)

        # image information
        self.imgInfo = QLabel("Image")
        main_layout.addWidget(self.imgInfo)

        # rrc switch
        hlayout = QHBoxLayout()
        self.rrcbtn = QPushButton("RRC On")
        self.rrcbtn.setMinimumHeight(40)
        self.rrcbtn.setCheckable(True)
        hlayout.addWidget(self.rrcbtn)

        # re-
        self.regeneratebtn = QPushButton("ReGenerate")
        self.regeneratebtn.setMinimumHeight(40)
        hlayout.addWidget(self.regeneratebtn)
        main_layout.addLayout(hlayout)

        # image show
        self.imageCanvas = ImageCanvas()
        # self.imageCanvas.showImage(loadTestImg())
        main_layout.addWidget(self.imageCanvas)

        # actions
        # def refresh(self,x=None):
        #
        # self.dataSetLen.refresh=refresh

        self.dataSetSelect.currentIndexChanged.connect(self.dataSetChange)
        self.open.clicked.connect(self.openSelect)
        self.next.clicked.connect(self.indexNext)
        self.back.clicked.connect(self.indexBack)
        self.index.returnPressed.connect(self.imageChange)

        # self.rrcbtn.clicked.connect(lambda :self.rrcbtn.setChecked(not self.rrcbtn.isChecked()))
        self.regeneratebtn.clicked.connect(self.imageChange)
        self.dataSetChange()
        # self.dataSetLen.set()

    def dataSetChange(self):
        t = self.dataSetSelect.currentText()
        if t == "Customized Folder":
            self.open.show()
            self.next.show()
            self.back.show()
            self.index.show()
            self.dataSetLen.setText(f"Please select folder")

        elif t == "Customized Image":
            self.open.show()
            self.next.hide()
            self.back.hide()
            self.index.hide()
            self.dataSetLen.setText(f"Image")
        else:
            self.dataSet = self.dataSets[t]()
            self.open.hide()
            self.next.show()
            self.back.show()
            self.index.show()
            self.dataSetLen.setText(f"Images Index From 0 to {len(self.dataSet) - 1}")
            self.index.setText("0")
            self.imageChange()

    def openSelect(self):
        t = self.dataSetSelect.currentText()
        if t == "Customized Folder":
            directory = QFileDialog.getExistingDirectory(directory="./")
            if directory:
                subdir = [entry for entry in os.scandir(directory) if entry.is_dir()]
                if not subdir:
                    self.dataSet = OnlyImages(directory)
                else:
                    self.dataSet = tv.datasets.ImageFolder(directory)
                self.dataSetLen.setText(f"Images Index From 0 to {len(self.dataSet) - 1}")
                self.index.setText("0")
                self.imageChange()

        elif t == "Customized Image":
            filename_long, f_type = QFileDialog.getOpenFileName(directory="./")
            if filename_long:
                self.img = Image.open(filename_long).convert('RGB')
                # self.img = np.asarray(img_PIL)
                self.imgInfo.setText(filename_long)
                self.imageChange()
        else:
            raise Exception()

    def indexNext(self):
        i = self.checkIndex()
        i += 1
        self.index.setText(str(self.checkIndex(i)))
        self.imageChange()

    def indexBack(self):
        i = self.checkIndex()
        i -= 1
        self.index.setText(str(self.checkIndex(i)))
        self.imageChange()

    def imageChange(self):
        t = self.dataSetSelect.currentText()
        if t == "Customized Folder":
            if self.dataSet is None:
                return
            i = self.checkIndex()
            self.img = self.dataSet[i][0]
            self.imgInfo.setText(f"{self.dataSet.samples[i][0]},cls:{self.dataSet.samples[i][1]}")
        elif t == "Customized Image":
            pass
        elif t == "Discrim DataSet":
            i = self.checkIndex()
            self.img = self.dataSet[i][0]
            self.imgInfo.setText(f"{self.dataSet.ds[i][0]},cls:{self.dataSet.ds[i][1]}")
        else:
            # gen img is tensor
            i = self.checkIndex()
            self.img = self.dataSet[i][0]
            self.imgInfo.setText(f"{self.dataSet.samples[i][0]},cls:{self.dataSet.samples[i][1]}")
        self.imageCanvas.showImage(self.img)
        if self.rrcbtn.isChecked():
            x = pilToRRCTensor(self.img)
        else:
            x = pilToTensor(self.img)
        x = toStd(x)
        x = x.unsqueeze(0)
        self.emitter.emit(x)

    def checkIndex(self, i=None):
        if self.dataSet is None or self.img is None:
            return 0
        try:
            if i is None:
                i = int(self.index.text())
            if i < 0 or i >= len(self.dataSet):
                i = 0
        except Exception as e:
            return 0
        return i

    def bindEmitter(self, signal: pyqtSignal):
        self.emitter = signal


class Predictor(QGroupBox):
    pass


class ExplainMethodSelector(QGroupBox):
    def __init__(self):
        super(ExplainMethodSelector, self).__init__()

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        # self.setMaximumWidth(600)
        self.setTitle("Explain Method Selector")

        self.models = {
            # "None": lambda: None,
            "vgg16": lambda: tv.models.vgg16(weights=tv.models.VGG16_Weights.DEFAULT),
            "alexnet": lambda: tv.models.alexnet(weights=tv.models.AlexNet_Weights.DEFAULT),
            "googlenet": lambda: tv.models.googlenet(weights=tv.models.GoogLeNet_Weights.DEFAULT),
            "resnet18": lambda: tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT),
            "resnet50": lambda: tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT),
        }
        self.model = None

        # partial fun 参数是静态的，传了就不能变，此处要求每次访问self.model。（写下语句的时候就创建完了）
        # lambda fun 是动态的，运行时解析
        # 结合一下匿名lambda函数就可以实现 创建含动态参数(model)的partial fun，只多了一步调用()
        cam_model_dict_by_layer = lambda model, layer: {'type': self.modelSelect.currentText(), 'arch': model,
                                                        'layer_name': layer, 'input_size': (224, 224)}
        interpolate_to_imgsize = lambda x: normalize_R(nf.interpolate(x.sum(1, True), 224, mode='bilinear'))
        multi_interpolate = lambda xs: normalize_R(
            sum(normalize_R(nf.interpolate(x.sum(1, True), 224, mode='bilinear')) for x in xs))

        # the method interface, all methods must follow this:
        # the method can call twice
        # 1. the method first accept "model" parameter, create a callable function "_m = m(model)"
        # 2. the heatmap generated by secondly calling "hm = _m(x,yc)"
        self.methods = {
            # "None": lambda model: None,
            # -----------CAM
            # --cam method using layer: 8,9,15,16,22,23,29,30
            "GradCAM-f": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, '-1')).__call__, sg=False,
                                               relu=True),  # cam can not auto release, so use partial
            "GradCAM-origin-f": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, '-1')).__call__, sg=False,
                                                      relu=False),
            "SG-GradCAM-origin-f": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, '-1')).__call__,
                                                         sg=True, relu=False),
            # "GradCAM-origin-29": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, '29')).__call__,sg=False, relu=False),
            # "SG-GradCAM-origin-29": lambda model: partial(GradCAM(cam_model_dict_by_layer(model, '29')).__call__,sg=True, relu=False),
            # GradCAM 23 is nonsense

            # --LayerCAM-origin-f == LRP-0-f
            # "LayerCAM-origin-f": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '-1')).__call__,
            #                                            sg=False, relu_weight=False, relu=False),
            # "LRP-0-f-grad": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_0(model, x, y, Relevance_Propagate=False)[31]),
            # "LRP-0-f-relev": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_0(model, x, y, Relevance_Propagate=True)[31]),
            # "LayerCAM-f": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '-1')).__call__,
            #                                     sg=False, relu_weight=True, relu=True),
            # "LayerCAM-origin-29": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '29')).__call__,sg=False, relu=False),
            # "LayerCAM-origin-23": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '23')).__call__,sg=False, relu=False),
            # "LayerCAM-origin-22": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '22')).__call__,sg=False, relu=False),
            # --SG LayerCAM-origin-f == ST-LRP-0-f
            # "SG-LayerCAM-origin-f": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '-1')).__call__,sg=True, relu=False),

            # "SG-LayerCAM-origin-29": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '29')).__call__,
            #                                   sg=True, relu=False),
            # "SG-LayerCAM-origin-23": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '23')).__call__,
            #                                   sg=True, relu=False),
            # "SG-LayerCAM-origin-22": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '22')).__call__,
            #                                   sg=True, relu=False),
            # "SG-LayerCAM-origin-16": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '16')).__call__,
            #                                                sg=True, relu=False),
            # "SG-LayerCAM-origin-0": lambda model: partial(LayerCAM(cam_model_dict_by_layer(model, '0')).__call__,
            #                                   sg=True, relu=False),

            # --others
            "ScoreCAM-f": lambda model: lambda x, y: ScoreCAM(model, '-1')(x, y, sg=True, relu=False),
            "AblationCAM-f": lambda model: lambda x, y: AblationCAM(model, -1)(x, y, relu=False),
            "RelevanceCAM-f": lambda model: lambda x, y: interpolate_to_imgsize(
                RelevanceCAM(model)(x, y, backward_init='c', method='lrpzp', layer=-1)),
            # "RelevanceCAM-24": lambda model: lambda x, y: interpolate_to_imgsize(
            #     RelevanceCAM(model)(x, y, backward_init='c', method='lrpzp', layer=24)),
            # "RelevanceCAM-1": lambda model: lambda x, y: interpolate_to_imgsize(
            #     RelevanceCAM(model)(x, y, backward_init='c', method='lrpzp', layer=1)),
            # "Taylor-30": lambda model:lambda x, y: interpolate_to_imgsize(Taylor(model, 30)(x, y)),

            # ------------LRP Top
            # # LRP-C use LRP-0 in classifier
            "LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='normal', method='lrpc', layer=-1)),
            # # LRP-Z is nonsense
            # "LRP-Z-f": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpz', layer=-1)
            #     .sum(1, True)),# lrpz 31 is bad
            # # LRP-ZP no edge highlight
            "LRP-ZP-f": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='normal', method='lrpzp', layer=-1)),
            # # LRP-W2 all red
            # "LRP-W2-f": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpw2', layer=-1)
            #     .sum(1, True)),
            "SIG0-LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=-1)),
            "SIGP-LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='sigp', method='lrpc', layer=-1)),
            "SG-LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer=-1)),
            "ST-LRP-C-f": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer=-1)),

            # "SIG0-LRP-ZP-f": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='sig0', method='lrpzp', layer=-1)),
            # "SIGP-LRP-ZP-f": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='sigp', method='lrpzp', layer=-1)),
            # "SG-LRP-ZP-31": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp', layer=31).sum(1, True)),
            # "ST-LRP-ZP-31": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='st', method='lrpzp', layer=31).sum(1, True)),
            # # to bad often loss discrimination
            # "C-LRP-C 31": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='c', method='lrpc', layer=31).sum(1, True)),
            # "C-LRP-ZP 31": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='c', method='lrpzp', layer=31).sum(1, True)),

            # ---------LRP-middle
            # "LRP C 30": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpc',layer=30).sum(1, True)),
            # "LRP C 24": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpc', layer=24).sum(1, True)),
            # "LRP C 23": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpc')
            #     [23].sum(1, True), 224, mode='bilinear'),
            # "SG LRP C 30": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpc',layer=30).sum(1, True),
            #     224, mode='bilinear'),
            # "SG LRP C 24": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpc')
            #     [24].sum(1, True), 224, mode='bilinear'),
            # "SG LRP C 23": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpc')
            #     [23].sum(1, True), 224, mode='bilinear'),
            # "ST LRP C 24": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='st', method='lrpc')
            #     [24].sum(1, True), 224, mode='bilinear'),
            # "ST LRP C 23": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='st', method='lrpc')
            #     [23].sum(1, True), 224, mode='bilinear'),

            "SIG0-LRP-C-24": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=24)),
            "SIG0-LRP-C-17": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=17)),
            "SIG0-LRP-C-10": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=10)),
            "SIG0-LRP-C-5": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=5)),

            # "LRP ZP 31": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpzp')
            #     [31].sum(1, True), 224, mode='bilinear'),
            # "LRP ZP 30": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpzp')
            #     [30].sum(1, True), 224, mode='bilinear'),
            # "LRP ZP 24": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpzp')
            #     [24].sum(1, True), 224, mode='bilinear'),
            # "LRP ZP 23": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='normal', method='lrpzp')
            #     [23].sum(1, True), 224, mode='bilinear'),
            # "SG LRP ZP 31": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp')
            #     [31].sum(1, True), 224, mode='bilinear'),
            # "SG LRP ZP 30": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp')
            #     [30].sum(1, True), 224, mode='bilinear'),
            # "SG LRP ZP 24": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp')
            #     [24].sum(1, True), 224, mode='bilinear'),
            # "SG LRP ZP 23": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='sg', method='lrpzp')
            #     [23].sum(1, True), 224, mode='bilinear'),
            # "ST LRP ZP 31": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='st', method='lrpzp')
            #     [31].sum(1, True), 224, mode='bilinear'),
            # "ST LRP ZP 30": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='st', method='lrpzp')
            #     [30].sum(1, True), 224, mode='bilinear'),
            # "ST LRP ZP 24": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='st', method='lrpzp')
            #     [24].sum(1, True), 224, mode='bilinear'),
            # "ST LRP ZP 23": lambda model: lambda x, y: nf.interpolate(
            #     LRP_Generator(model)(x, y, backward_init='st', method='lrpzp')
            #     [23].sum(1, True), 224, mode='bilinear'),

            # ----------LRP-pixel
            "LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='normal', method='lrpc', layer=1)),
            "LRP-C-0": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='normal', method='lrpc', layer=0)),
            "SG-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer=1)),
            "SG-LRP-C-0": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='sg', method='lrpc', layer=0)),
            "ST-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer=1)),
            "ST-LRP-C-0": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='st', method='lrpc', layer=0)),
            "SIG0-LRP-C-1": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=1)),
            "SIG0-LRP-C-0": lambda model: lambda x, y: interpolate_to_imgsize(
                LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=0)),
            # # noisy
            # "LRP-0 0": lambda model: lambda x, y: LRP_Generator(model)(
            #     x, y, backward_init='normal', method='lrp0', layer=0).sum(1, True),
            # # nonsense
            # "LRP-Z 0": lambda model: lambda x, y: LRP_Generator(model)(
            #     x, y, backward_init='normal', method='lrpz', layer=0).sum(1, True),
            # # noisy
            # "S-LRP-C 1": lambda model: lambda x, y: LRP_Generator(model)(
            #     x, y, backward_init='normal', method='slrp', layer=1).sum(1, True),
            # "S-LRP-C 0": lambda model: lambda x, y: LRP_Generator(model)(
            #     x, y, backward_init='normal', method='slrp', layer=0).sum(1, True),
            # "LRP-ZP 0": lambda model: lambda x, y: LRP_Generator(model)(
            #     x, y, backward_init='normal', method='lrpzp', layer=0).sum(1, True),
            # "SG-LRP-ZP 0": lambda model: lambda x, y: LRP_Generator(model)(
            #     x, y, backward_init='sg', method='lrpzp', layer=0).sum(1, True),

            # IG
            "IG": lambda model: lambda x, y: interpolate_to_imgsize(
                IGDecomposer(model)(x, y, post_softmax=False)),
            "SIG": lambda model: lambda x, y: interpolate_to_imgsize(
                IGDecomposer(model)(x, y, post_softmax=True)),

            # -----------Increment Decomposition
            # LID-linear?-init-middle-end.
            # LID-Taylor-sig-f means it is layer linear decompose, given sig init , ending at feature layer
            # LID-IG-sig-1 means it is layer integrated decompose, given sig init , ending at layer-1
            "LID-Taylor-f": lambda model: lambda x, y: interpolate_to_imgsize(
                LIDLinearDecomposer(model)(x, y, layer=-1, Relevance_Propagate=False)),
            # "LID-Taylor-f-relev": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDLinearDecomposer(model)(x, y, layer=-1, Relevance_Propagate=True)),
            "LID-Taylor-sig-f": lambda model: lambda x, y: interpolate_to_imgsize(
                LIDLinearDecomposer(model)(x, y, layer=-1, backward_init='sig')),

            "LID-IG-f": lambda model: lambda x, y: interpolate_to_imgsize(
                LIDIGDecomposer(model)(x, y, layer=-1, backward_init='normal')),
            "LID-IG-sig-f": lambda model: lambda x, y: interpolate_to_imgsize(
                LIDIGDecomposer(model)(x, y, layer=-1, backward_init='sig')),

            # "LID-Taylor-1": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDLinearDecomposer(model)(x, y, layer=1)),
            # "LID-Taylor-sig-1": lambda model: lambda x, y: interpolate_to_imgsize(
            #     LIDLinearDecomposer(model)(x, y, layer=1, backward_init='sig')),

            "LID-IG-1": lambda model: lambda x, y: interpolate_to_imgsize(
                LIDIGDecomposer(model)(x, y, layer=1, backward_init='normal')),
            "LID-IG-sig-1": lambda model: lambda x, y: interpolate_to_imgsize(
                LIDIGDecomposer(model)(x, y, layer=1, backward_init='sig')),

            "LID-IG-sig-24": lambda model: lambda x, y: interpolate_to_imgsize(
                LIDIGDecomposer(model)(x, y, layer=24, backward_init='sig')),
            "LID-IG-sig-17": lambda model: lambda x, y: interpolate_to_imgsize(
                LIDIGDecomposer(model)(x, y, layer=17, backward_init='sig')),
            "LID-IG-sig-10": lambda model: lambda x, y: interpolate_to_imgsize(
                LIDIGDecomposer(model)(x, y, layer=10, backward_init='sig')),
            "LID-IG-sig-5": lambda model: lambda x, y: interpolate_to_imgsize(
                LIDIGDecomposer(model)(x, y, layer=5, backward_init='sig')),

            # mix methods
            "SIG0-LRP-C-m": lambda model: lambda x, y: multi_interpolate(
                hm for i, hm in enumerate(LRP_Generator(model)(x, y, backward_init='sig0', method='lrpc', layer=None))
                if i in [1, 5, 10, 17, 24]),
            "LID-Taylor-sig-m": lambda model: lambda x, y: multi_interpolate(
                hm for i, hm in enumerate(LIDLinearDecomposer(model)(x, y, layer=None, backward_init='sig'))
                if i in [24, 31]),
            "LID-IG-sig-m": lambda model: lambda x, y: multi_interpolate(
                hm for i, hm in enumerate(LIDIGDecomposer(model)(x, y, layer=None, backward_init='sig'))
                if i in [1, 5, 10, 17, 24]),


        }

        # the mask interface, all masks must follow this:
        # the masked heatmap generated by calling "im, cover = m(hm, im)"
        # the param hm is raw heatmap, the im is input image
        # the output im is a valid printable masked heatmap image

        self.masks = {
            "Raw Heatmap": lambda hm, im: (None, hm),
            "Overlap": lambda hm, im: (invStd(im), hm),
            "Positive Only": lambda hm, im: (invStd(im * positize(hm)), None),
            "Sparsity 50": lambda hm, im: (invStd(im * binarize(hm, sparsity=0.5)), None),
            "Maximal Patch": lambda hm, im: (invStd(im * maximalPatch(hm, top=True, r=10)), None),
            "Corner Mask": lambda hm, im: (invStd(im * cornerMask(hm, r=40)), None),
            "AddNoiseN 0.5": lambda hm, im: (invStd(im + binarize_add_noisy_n(hm, top=True, std=0.5)), None),
            "AddNoiseN 1": lambda hm, im: (invStd(im + binarize_add_noisy_n(hm, top=True, std=1)), None),
            "AddNoiseN 0.5 Inv": lambda hm, im: (invStd(im + binarize_add_noisy_n(hm, top=False, std=0.5)), None),
            "AddNoiseN 1 Inv": lambda hm, im: (invStd(im + binarize_add_noisy_n(hm, top=False, std=1)), None),
        }

        self.method = None
        self.img = None

        self.mask = None

        hlayout = QHBoxLayout()
        self.modelSelect = QComboBox()
        temp = QStandardItemModel()
        for key in self.models:
            temp2 = QStandardItem(key)
            temp2.setData(key)
            temp2.setSizeHint(QSize(200, 40))
            temp.appendRow(temp2)
        self.modelSelect.setModel(temp)
        self.modelSelect.setCurrentIndex(0)
        self.modelSelect.setMinimumHeight(40)
        hlayout.addWidget(TipedWidget("Model: ", self.modelSelect))

        self.methodSelect = QComboBox()
        temp = QStandardItemModel()
        for key in self.methods:
            temp2 = QStandardItem(key)
            temp2.setData(key)
            temp2.setSizeHint(QSize(200, 40))
            temp.appendRow(temp2)
        self.methodSelect.setModel(temp)
        self.methodSelect.setCurrentIndex(0)
        self.methodSelect.setMinimumHeight(40)
        hlayout.addWidget(TipedWidget("Method: ", self.methodSelect))
        main_layout.addLayout(hlayout)
        del hlayout

        hlayout = QHBoxLayout()
        self.maskSelect = QComboBox()
        temp = QStandardItemModel()
        for key in self.masks:
            temp2 = QStandardItem(key)
            temp2.setData(key)
            temp2.setSizeHint(QSize(200, 40))
            temp.appendRow(temp2)
        self.maskSelect.setModel(temp)
        self.maskSelect.setCurrentIndex(1)
        self.maskSelect.setMinimumHeight(40)
        hlayout.addWidget(TipedWidget("Mask: ", self.maskSelect))

        self.regeneratebtn1 = QPushButton("Get Image")
        self.regeneratebtn1.setMinimumHeight(40)
        hlayout.addWidget(self.regeneratebtn1)
        # re-
        self.regeneratebtn2 = QPushButton("ReGenerate Heatmap")
        self.regeneratebtn2.setMinimumHeight(40)
        hlayout.addWidget(self.regeneratebtn2)
        main_layout.addLayout(hlayout)
        del hlayout

        # class
        self.classSelector = QLineEdit("")
        self.classSelector.setMinimumHeight(40)
        # self.classSelector.setMaximumWidth(500)
        self.classSelector.setPlaceholderText("classes choose:")
        main_layout.addWidget(TipedWidget("Classes:", self.classSelector))

        # class sort
        self.topk = 10
        main_layout.addWidget(QLabel(f"Prediction Top {self.topk}"))
        self.predictionScreen = QPlainTextEdit("this is place classes predicted shown\nexample:class 1 , Cat")
        self.predictionScreen.setMinimumHeight(40)
        # self.predictionScreen.setMaximumWidth(800)
        self.predictionScreen.setReadOnly(True)
        main_layout.addWidget(self.predictionScreen)

        # output canvas
        self.maxHeatmap = 6
        self.imageCanvas = ImageCanvas()  # no add

        # actions
        self.modelSelect.currentIndexChanged.connect(self.modelChange)
        self.methodSelect.currentIndexChanged.connect(self.methodChange)
        self.maskSelect.currentIndexChanged.connect(self.maskChange)

        self.regeneratebtn1.clicked.connect(self.callImgChg)
        self.classSelector.returnPressed.connect(self.HeatMapChange)
        self.regeneratebtn2.clicked.connect(self.HeatMapChange)

    def init(self):
        # default calling
        self.modelChange()
        self.methodChange()
        self.maskChange()

    # def outputCanvasWidget(self):
    #     return self.imageCanvas

    def modelChange(self):
        t = self.modelSelect.currentText()
        self.model = self.models[t]()
        if self.model is not None:
            self.model = self.model.eval().to(device)
            t = self.methodSelect.currentText()
            self.method = self.methods[t](self.model)
            self.ImageChange()

    def methodChange(self):
        if self.model is not None:
            t = self.methodSelect.currentText()
            self.method = self.methods[t](self.model)
            self.HeatMapChange()

    def maskChange(self):
        if self.method is None:
            return
        t = self.maskSelect.currentText()
        self.mask = self.masks[t]
        self.HeatMapChange()

    def bindReciever(self, signal):
        self.reciever = signal
        self.reciever.connect(self.ImageChange)

    def saveImgnt(self, imgnt):
        self.imgnt = imgnt
        self.classes = loadImageNetClasses()

    def ImageChange(self, x=None):
        if x is not None:
            self.img = x
        if self.img is None:
            return
        # 测试，输出传入的图像
        # self.imageCanvas.showImage(ToPlot(InverseStd(self.img)))
        if self.model is not None:
            # predict
            self.img_dv = self.img.to(device)
            prob = tc.softmax(self.model(self.img_dv), 1)
            topki = prob.sort(1, True)[1][0, :self.topk]
            topkv = prob.sort(1, True)[0][0, :self.topk]
            # show info
            if self.classes is None:
                raise Exception("saveClasses First.")
            self.predictionScreen.setPlainText(
                "\n".join(self.PredInfo(i.item(), v.item()) for i, v in zip(topki, topkv))
            )
            self.classSelector.setText(",".join(str(i.item()) for i in topki))
            if self.method is not None:
                self.HeatMapChange()

    def PredInfo(self, cls, prob=None):
        if prob:
            s = f"{cls}:\t{prob:.4f}:\t{self.classes[cls]}"
        else:
            s = f"{cls}:{self.classes[cls]}"
        return s

    def HeatMapChange(self):
        if self.img is None or self.model is None or self.method is None:
            return
        try:
            # get classes
            classes = list(int(cls) for cls in self.classSelector.text().split(','))
            classes = classes[:self.maxHeatmap]  # always 6
            # classes = classes[:6]
        except Exception as e:
            self.predictionScreen.setPlainText(e.__str__())
            return
        # heatmaps
        if USING_DRAW_BACKEND == 'mpl':
            # self.imageCanvas.clear()
            # 必须用对象自带的figure，否则大小不能自动调整
            fig = self.imageCanvas.figure
            fig.clf()
            # 显存有时候会超限
            # tc.cuda.empty_cache()
            fig.tight_layout()
            # fig.subplots(3,2)
            # 创建多张图，返回list[figure]，在返回一张的时候只返回figure
            sfs = fig.subfigures(ceil(len(classes) / 2), min(len(classes), 2))
            # 当然每张图分为左右部分，左边是热力图，右边是例子
            # ___________runningCost___________ = RunningCost(50)
            for i, cls in enumerate(classes):
                # ___________runningCost___________.tic()
                sf = sfs.flatten()[i] if len(classes) > 1 else sfs
                # sf.set_title(str(cls))
                # sf.tight_layout()
                # axe=fig.add_subplot(3,2,i+1)
                hm = self.method(self.img_dv, cls).detach().cpu()
                # ___________runningCost___________.tic("generate heatmap")
                wnid = self.imgnt.wnids[cls]
                directory = self.imgnt.split_folder
                target_dir = os.path.join(directory, wnid)
                if not os.path.isdir(target_dir):
                    raise Exception()
                allImages = list(e.name for e in os.scandir(target_dir))
                imgCount = len(allImages)
                j = random.randint(0, imgCount - 1)
                imgpath = os.path.join(target_dir, allImages[j])
                # allImages=[]
                # for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                #     for fname in sorted(fnames):
                #         path = os.path.join(root, fname)
                #         allImages.append(path)
                example = Image.open(imgpath).convert("RGB")
                # ___________runningCost___________.tic("prepare example")
                axe = sf.add_subplot(1, 2, 1)
                p = None
                if self.mask is not None:
                    masked, covering = self.mask(self.img, hm)
                    if masked is not None:
                        p = self.maskScore(masked, cls)
                        axe.imshow(toPlot(masked), cmap=None, vmin=0, vmax=1)
                        if covering is not None:
                            axe.imshow(toPlot(covering), cmap=lrp_cmap, alpha=0.7, vmin=-1, vmax=1)
                    elif covering is not None:
                        axe.imshow(toPlot(covering), cmap=lrp_cmap, vmin=-1, vmax=1)
                else:
                    axe.imshow(toPlot(normalize_R(hm)), cmap=lrp_cmap, vmin=-1, vmax=1)
                axe.set_axis_off()
                axe.set_title(self.PredInfo(cls, p))
                axe = sf.add_subplot(1, 2, 2)
                axe.imshow(example)
                axe.set_axis_off()
                # pair=[hm,example]
                # hmpair.append(pair)
                # time of plotting chasing time of generating heatmap. any alternative?
            #     ___________runningCost___________.tic("ploting")
            # ___________runningCost___________.cost()
            self.imageCanvas.showFigure(fig)

        elif USING_DRAW_BACKEND == 'gl':
            self.imageCanvas.pglw.clear()
            # ___________runningCost___________ = RunningCost(20)
            row = -1
            col = -1
            for i, cls in enumerate(classes):
                # ___________runningCost___________.tic()
                col += 1
                if i % 2 == 0:
                    row += 1
                    col = 0
                    # self.imageCanvas.pglw.nextRow()
                l = self.imageCanvas.pglw.addLayout(row=row, col=col)  # 2 images
                hm = self.method(self.img_dv, cls).detach().cpu()
                # ___________runningCost___________.tic("generate heatmap")
                wnid = self.imgnt.wnids[cls]
                directory = self.imgnt.split_folder
                target_dir = os.path.join(directory, wnid)
                if not os.path.isdir(target_dir):
                    raise Exception()
                allImages = list(e.name for e in os.scandir(target_dir))
                imgCount = len(allImages)
                j = random.randint(0, imgCount - 1)
                imgpath = os.path.join(target_dir, allImages[j])
                example = pilOpen(imgpath)
                # ___________runningCost___________.tic("prepare example")
                pim: pg.PlotItem = l.addPlot(0, 0)
                p = None
                if self.mask is not None:
                    masked, covering = self.mask(hm, self.img)
                    if masked is not None:
                        p = self.maskScore(masked, cls)
                        if covering is not None:
                            pim.addItem(pg.ImageItem(toPlot(masked).numpy(), opacity=1.))
                            pim.addItem(pg.ImageItem(toPlot(covering).numpy(),
                                                     # compositionMode=QPainter.CompositionMode.CompositionMode_Overlay,
                                                     levels=[-1, 1], lut=lrp_cmap_gl.getLookupTable(), opacity=0.7))
                        else:
                            pim.addItem(pg.ImageItem(toPlot(masked).numpy()))

                    elif covering is not None:
                        pim.addItem(
                            pg.ImageItem(toPlot(covering).numpy(), levels=[-1, 1], lut=lrp_cmap_gl.getLookupTable()))
                else:
                    pim.addItem(pg.ImageItem(toPlot(hm).numpy(), levels=[-1, 1], lut=lrp_cmap_gl.getLookupTable()))
                plotItemDefaultConfig(pim)
                pexp: pg.PlotItem = l.addPlot(0, 1)
                im_exp = toPlot(pilToTensor(example)).numpy()
                # hw=min(im_exp.shape[0],im_exp.shape[1])
                # pexp.setFixedWidth(500)
                # pexp.setFixedHeight(500)
                pexp.addItem(pg.ImageItem(im_exp))
                pexp.setTitle(self.PredInfo(cls, p))
                plotItemDefaultConfig(pexp)
                # l.setStretchFactor(pim,1)
                # l.setStretchFactor(pexp,1)
                # ___________runningCost___________.tic("ploting")
            # ___________runningCost___________.cost()

    def maskScore(self, x, y):
        if self.maskSelect.currentText() in ["Positive Only", "Sparsity 50", "Maximal Patch", "Corner Mask",
                                             "AddNoiseN 0.5", "AddNoiseN 1", "AddNoiseN 0.5 Inv", "AddNoiseN 1 Inv"]:
            return self.getProb(x, y)
        else:
            return None

    def getProb(self, x, y):
        return self.model(toStd(x).to(device)).softmax(1)[0, y]

    def callImgChg(self):
        # 我真的服，signal擅自修改了我的默认参数。这下你添加不了了吧
        self.ImageChange()


class MainWindow(QMainWindow):
    # 信号应该定义在类中
    imageChangeSignal = pyqtSignal(tc.Tensor)

    def __init__(self):
        super().__init__()

        ### set mainFrame UI
        ##  main window settings
        # self.setGeometry(200, 100, 1000, 800)  # this is nonsense
        # self.frameGeometry().moveCenter(QDesktopWidget.availableGeometry().center())
        self.setWindowTitle("Explaining Visualization")
        # self.setWindowIcon(QIcon('lidar.ico'))
        # self.setIconSize(QSize(20, 20))

        mainPanel = QWidget()
        self.setCentralWidget(mainPanel)
        control_layout = QHBoxLayout()
        cleft_panel = QVBoxLayout()
        cright_panel = QVBoxLayout()
        mainPanel.setLayout(control_layout)
        control_layout.addLayout(cleft_panel)
        control_layout.addLayout(cright_panel)
        # 划分屏幕为左右区域，以cleft_panel 和 cright_panel来添加垂直控件。

        # 左屏幕
        self.imageLoader = ImageLoader()
        cleft_panel.addWidget(self.imageLoader)
        self.imageLoader.bindEmitter(self.imageChangeSignal)
        self.explainMethodsSelector = ExplainMethodSelector()
        cleft_panel.addWidget(self.explainMethodsSelector)
        self.explainMethodsSelector.saveImgnt(self.imageLoader.imageNetVal)
        self.explainMethodsSelector.bindReciever(self.imageChangeSignal)
        self.explainMethodsSelector.init()
        # 右屏幕
        cright_panel.addWidget(self.explainMethodsSelector.imageCanvas)
        # Show
        control_layout.setStretchFactor(cleft_panel, 1)
        control_layout.setStretchFactor(cright_panel, 4)
        # cleft_panel.setStretchFactor(1)
        # cright_panel.setStretchFactor(4)
        # self.imageLoader.setMaximumWidth(800)
        # self.explainMethodsSelector.setMaximumWidth(800)
        self.showMaximized()


def main():
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
