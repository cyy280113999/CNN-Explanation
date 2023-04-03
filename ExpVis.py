# sys
import os
import sys
import random
from functools import partial
from math import ceil
# nn
import numpy as np
import torch as tc
import torch.nn.functional as nf
import torchvision as tv
# gui
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QPainter
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QGroupBox, QVBoxLayout, QComboBox, QPushButton, QLineEdit, \
    QFileDialog, QMainWindow, QApplication

# user
from utils import *
from datasets.OnlyImages import OnlyImages
from datasets.DiscrimDataset import *

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

    pyqtgraphDefaultConfig()
    # import pyqtgraph.opengl as gl
    import pyqtgraph.colormap as pcolors

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


imageNetVal = getImageNet('val', None)


class ImageLoader(QGroupBox):
    def __init__(self):
        super().__init__()
        # key: dataset name , value: is folder or not
        self.classes = loadImageNetClasses()
        self.dataSets = {
            "Customized Image": None,
            "Customized Folder": None,
            "ImageNet Val": lambda: imageNetVal,
            "ImageNet Train": lambda: getImageNet('train', None),
            "Discrim DataSet": lambda: DiscrimDataset(transform=None, MultiLabel=False),
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
        self.randbtn = QPushButton("Rand")
        self.index = QLineEdit("0")
        hlayout.addWidget(self.open)
        hlayout.addWidget(self.back)
        hlayout.addWidget(self.next)
        hlayout.addWidget(self.randbtn)
        hlayout.addWidget(self.index)
        main_layout.addLayout(hlayout)

        # self.open.setFixedSize(80,40)
        # self.back.setFixedSize(80,40)
        # self.next.setFixedSize(80,40)
        self.open.setMinimumHeight(40)
        self.back.setMinimumHeight(40)
        self.next.setMinimumHeight(40)
        self.randbtn.setMinimumHeight(40)
        self.index.setMinimumHeight(40)
        self.index.setMaximumWidth(80)
        self.index.setMaxLength(8)

        # image information
        self.imgInfo = QLabel("Image")
        main_layout.addWidget(self.imgInfo)

        # new line
        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel('image modify:'))
        # rrc switch
        self.rrcbtn = QPushButton("RRC")
        self.rrcbtn.setMinimumHeight(40)
        self.rrcbtn.setCheckable(True)
        hlayout.addWidget(self.rrcbtn)
        # image modify after tostd
        # interface: im->im
        self.modes = {
            "None": None,
            # "Positive Only": lambda hm, im: (invStd(im * positize(hm)), None),
            # "Sparsity 50": lambda hm, im: (invStd(im * binarize(im, sparsity=0.5)), None),
            "Corner Mask": lambda im: im * cornerMask(im, r=40),
            "AddNoise 0.1Std": lambda im: im + 0.1 * torch.randn_like(im),
            "AddNoise 0.3Std": lambda im: im + 0.3 * torch.randn_like(im),
            "AddNoise 0.5Std": lambda im: im + 0.5 * torch.randn_like(im),
        }
        self.imageMode = None
        self.modeSelects = QComboBox()
        temp = QStandardItemModel()
        for key in self.modes:
            temp2 = QStandardItem(key)
            temp2.setData(key)
            temp2.setSizeHint(QSize(200, 40))
            temp.appendRow(temp2)
        self.modeSelects.setModel(temp)
        self.modeSelects.setCurrentIndex(0)
        self.modeSelects.setMinimumHeight(40)
        hlayout.addWidget(self.modeSelects)

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
        self.randbtn.clicked.connect(self.indexRand)
        self.index.returnPressed.connect(self.imageChange)

        # self.rrcbtn.clicked.connect(lambda :self.rrcbtn.setChecked(not self.rrcbtn.isChecked()))
        self.modeSelects.currentIndexChanged.connect(self.modeChange)
        self.regeneratebtn.clicked.connect(self.imageChange)
        self.dataSetChange()
        # self.dataSetLen.set()

    def dataSetChange(self):
        t = self.dataSetSelect.currentText()
        if t == "Customized Folder":
            self.open.show()
            self.next.show()
            self.back.show()
            self.randbtn.show()
            self.index.show()
            self.dataSetLen.setText(f"Please select folder")

        elif t == "Customized Image":
            self.open.show()
            self.next.hide()
            self.back.hide()
            self.randbtn.hide()
            self.index.hide()
            self.dataSetLen.setText(f"Image")
        else:
            self.dataSet = self.dataSets[t]()
            self.open.hide()
            self.next.show()
            self.back.show()
            self.randbtn.show()
            self.index.show()
            self.dataSetLen.setText(f"Images Index From 0 to {len(self.dataSet) - 1}")
            self.checkIndex(0)
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

    def checkIndex(self, i=None):
        if self.dataSet is None:
            return None
        if i is not None:
            if isinstance(i, str):
                i = int(i)
            i = i % len(self.dataSet)
            self.index.setText(str(i))
        else:
            i = int(self.index.text())
        return i

    def indexNext(self):
        i = self.checkIndex()
        self.checkIndex(i + 1)
        self.imageChange()

    def indexBack(self):
        i = self.checkIndex()
        self.checkIndex(i - 1)
        self.imageChange()

    def indexRand(self):
        i = torch.randint(0, len(self.dataSet) - 1, (1,)).item()
        self.checkIndex(i)
        self.imageChange()

    def modeChange(self):
        t = self.modeSelects.currentText()
        self.imageMode = self.modes[t]
        self.imageChange()

    def imageChange(self):
        t = self.dataSetSelect.currentText()
        if t == "Customized Folder":
            if self.dataSet is None:
                return
            i = self.checkIndex()
            self.img, _ = self.dataSet[i]
            path, cls = self.dataSet.samples[i]
            self.imgInfo.setText(f"{path},cls:{cls}")
        elif t == "Customized Image":
            pass
        elif t == "Discrim DataSet":
            i = self.checkIndex()
            self.img, _ = self.dataSet[i]
            path, cls = self.dataSet.ds[i]
            self.imgInfo.setText(f"{path},cls:{cls}")
        else:
            # gen img is tensor
            i = self.checkIndex()
            self.img, _ = self.dataSet[i]
            path, cls = self.dataSet.samples[i]
            self.imgInfo.setText(f"{path},cls:{cls}")
        if self.img is not None:
            self.img = toTensorS224(self.img)
            if self.rrcbtn.isChecked():
                self.img = toRRC(self.img)
            self.img = toStd(self.img)
            if self.imageMode:
                self.img = self.imageMode(self.img)
            self.imageCanvas.showImage(toPlot(invStd(self.img)).clip(min=0, max=1))
            self.img = self.img.unsqueeze(0)
            self.emitter.emit(self.img)

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
            "resnet34": lambda: tv.models.resnet34(weights=tv.models.ResNet34_Weights.DEFAULT),
            "resnet50": lambda: tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT),
        }
        self.model = None

        from HeatmapMethods import heatmap_methods
        self.methods = heatmap_methods
        # the mask interface, all masks must follow this:
        # the masked heatmap generated by calling "im, cover = m(hm, im)"
        # the param hm is raw heatmap, the im is input image
        # the output im is a valid printable masked heatmap image
        # the output hm is raw heatmap
        # mask: hm, im -> im, hm

        self.masks = {
            "Raw Heatmap": lambda hm, im: (None, hm),
            "Overlap": lambda hm, im: (invStd(im), hm),
            "Positive Only": lambda hm, im: (invStd(im * positize(hm)), None),
            "Sparsity 50": lambda hm, im: (invStd(im * binarize(hm, sparsity=0.5)), None),
            "Maximal Patch": lambda hm, im: (invStd(im * maximalPatch(hm, top=True, r=20)), None),
            "Minimal Patch": lambda hm, im: (invStd(im * maximalPatch(hm, top=False, r=20)), None),
            "Corner Mask": lambda hm, im: (invStd(im * cornerMask(hm, r=40)), None),
            # "AddNoise 0.1Std": lambda hm, im: (invStd(im + 0.1 * torch.randn_like(im)), None),
            # "AddNoise 0.5Std": lambda hm, im: (invStd(im + 0.5 * torch.randn_like(im)), None),
            "AddNoise By50%Hm N0.5Std": lambda hm, im: (invStd(im + binarize_add_noisy_n(hm, top=True, std=0.5)), None),
            "AddNoise By50%Hm N1Std": lambda hm, im: (invStd(im + binarize_add_noisy_n(hm, top=True, std=1)), None),
            "AddNoise ByInv50%Hm N0.5Std": lambda hm, im: (
            invStd(im + binarize_add_noisy_n(hm, top=False, std=0.5)), None),
            "AddNoise ByInv50%Hm N1Std": lambda hm, im: (invStd(im + binarize_add_noisy_n(hm, top=False, std=1)), None),
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
        self.maskSelect.setCurrentIndex(0)
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

    def setCanvas(self, canvas):
        # output canvas
        self.maxHeatmap = 6
        if canvas is not None:
            self.imageCanvas = canvas
        else:
            self.imageCanvas = ImageCanvas()  # no add

    # def outputCanvasWidget(self):
    #     return self.imageCanvas

    def modelChange(self):
        t = self.modelSelect.currentText()
        if self.models[t] is not None:
            self.model = self.models[t]().eval().to(device)
            self.methodChange()

    def methodChange(self):
        if self.model is not None:
            t = self.methodSelect.currentText()
            if self.methods[t] is not None:
                self.method = self.methods[t](self.model)
                self.HeatMapChange()

    def maskChange(self):
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
        # test, send img to canvas
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
                    axe.imshow(toPlot(heatmapNormalizeR(hm)), cmap=lrp_cmap, vmin=-1, vmax=1)
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
                example = self.cls_example(cls)
                im_exp = toPlot(toTensorS224(example))
                # ___________runningCost___________.tic("prepare example")
                pi: pg.PlotItem = l.addPlot(0, 0)
                p = None
                if self.mask is not None:
                    masked, covering = self.mask(hm, self.img)
                    if masked is not None:
                        p = self.maskScore(masked, cls)
                        if covering is not None:
                            pi.addItem(pg.ImageItem(toPlot(masked), opacity=1.))
                            pi.addItem(pg.ImageItem(toPlot(covering),
                                                    # compositionMode=QPainter.CompositionMode.CompositionMode_Overlay,
                                                    levels=[-1, 1], lut=lrp_lut, opacity=0.7))
                        else:
                            pi.addItem(pg.ImageItem(toPlot(masked)))

                    elif covering is not None:
                        pi.addItem(
                            pg.ImageItem(toPlot(covering), levels=[-1, 1], lut=lrp_lut))
                else:
                    pi.addItem(pg.ImageItem(toPlot(hm), levels=[-1, 1], lut=lrp_lut))
                plotItemDefaultConfig(pi)
                pexp: pg.PlotItem = l.addPlot(0, 1)
                # hw=min(im_exp.shape[0],im_exp.shape[1])
                # pexp.setFixedWidth(500)
                # pexp.setFixedHeight(500)
                pexp.addItem(pg.ImageItem(im_exp))
                pexp.setTitle(self.PredInfo(cls, p))
                plotItemDefaultConfig(pexp)
                # l.setStretchFactor(pi,1)
                # l.setStretchFactor(pexp,1)
                # ___________runningCost___________.tic("ploting")
            # ___________runningCost___________.cost()

    def cls_example(self, cls):
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
        return example

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
    # must define signal in class
    imageChangeSignal = pyqtSignal(tc.Tensor)

    def __init__(self, imageLoader, explainMethodsSelector, imageCanvas,
                 SeperatedCanvas=True):
        super().__init__()
        self.imageLoader = imageLoader
        self.explainMethodsSelector = explainMethodsSelector
        self.imageCanvas = imageCanvas

        # set mainFrame UI, main window settings
        self.setWindowTitle("Explaining Visualization")
        # self.setGeometry(200, 100, 1000, 800) #specify window size
        # self.frameGeometry().moveCenter(QDesktopWidget.availableGeometry().center())
        # self.setWindowIcon(QIcon('EXP.ico'))
        # self.setIconSize(QSize(20, 20))

        mainPanel = QWidget()
        self.setCentralWidget(mainPanel)
        control_layout = QHBoxLayout()
        # split window into L&R.
        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()
        mainPanel.setLayout(control_layout)
        control_layout.addLayout(left_panel)
        control_layout.addLayout(right_panel)
        if SeperatedCanvas:
            # add controllers
            left_panel.addWidget(self.imageLoader)
            right_panel.addWidget(self.explainMethodsSelector)
            self.imageCanvas.showMaximized()
            self.show()
        else:
            # left_panel add controllers
            left_panel.addWidget(self.imageLoader)
            left_panel.addWidget(self.explainMethodsSelector)
            # cright_panel add display screen
            right_panel.addWidget(self.imageCanvas)
            control_layout.setStretchFactor(left_panel, 1)
            control_layout.setStretchFactor(right_panel, 4)
            # left_panel.setStretchFactor(1)
            # right_panel.setStretchFactor(4)
            # self.imageLoader.setMaximumWidth(800)
            # self.explainMethodsSelector.setMaximumWidth(800)
            self.showMaximized()


def main(SeperatedWindow=True):
    global imageNetVal
    # --workflow
    # create window
    app = QApplication(sys.argv)
    imageLoader = ImageLoader()  # keep this instance alive!
    explainMethodsSelector = ExplainMethodSelector()
    imageCanvas = ImageCanvas()
    mw = MainWindow(imageLoader, explainMethodsSelector, imageCanvas, SeperatedCanvas=SeperatedWindow)
    # initial settings
    explainMethodsSelector.saveImgnt(imageNetVal)
    imageLoader.bindEmitter(mw.imageChangeSignal)
    explainMethodsSelector.bindReciever(mw.imageChangeSignal)
    explainMethodsSelector.init()
    explainMethodsSelector.setCanvas(imageCanvas)
    mw.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
