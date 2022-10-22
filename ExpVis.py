# sys
import os
import sys
import random
from functools import partial
# draw
from PIL import Image
import matplotlib as mpl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
# nn
import numpy as np
import torch as tc
import torchvision as tv
# gui
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QGroupBox, QVBoxLayout, QComboBox, QPushButton, QLineEdit, \
    QFileDialog, QMainWindow, QApplication
# user
from utils import InverseStd, ToPlot, transToStd, transToTensor, positize, maximalPatch, binarize, \
    normalize, normalize_R2P, normalize_R, lrp_cmap, RRCTensor, RunningCost
from OnlyImages import OnlyImages
from cam.gradcam import GradCAM
from cam.layercam import LayerCAM
from LRP import LRP_Generator

# mpl initialize
mpl.use('QtAgg')  # 指定渲染后端。QtAgg后端指用Agg二维图形库在Qt控件上绘图。
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

# torch initial
device = "cuda"


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
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        self.setTitle("Image:")
        self.figure = Figure()
        self.axe = None
        self.canvas = FigureCanvasQTAgg(self.figure)
        main_layout.addWidget(self.canvas)

    def showImage(self, img):
        self.figure.clf()
        self.axe = self.figure.add_subplot()
        self.axe.imshow(img)
        self.axe.set_axis_off()
        self.figure.tight_layout()
        self.canvas.draw_idle()  # 刷新

    def showFigure(self, fig):
        self.axe = None
        if self.figure is not fig:
            self.figure = fig
        self.canvas.figure = self.figure
        self.canvas.draw_idle()  # 刷新

    def clear(self):
        self.figure=None
        self.canvas.figure=None

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
        self.classes = self.imageNetVal.classes
        self.dataSets = {
            "Customized Image": None,
            "Customized Folder": None,
            "ImageNet Val": lambda: self.imageNetVal,
            "ImageNet Train": partial(tv.datasets.ImageNet, root="F:/DataSet/imagenet", split="train"),
        }
        self.dataSet = None
        self.img = None
        self.setMaximumWidth(600)
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
        hlayout=QHBoxLayout()
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
        hlayout=QHBoxLayout()
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
            self.dataSetLen.setText(f"Images Index From 0 to {len(self.dataSet)-1}")
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
        else:
            # gen img is tensor
            i = self.checkIndex()
            self.img = self.dataSet[i][0]
            self.imgInfo.setText(f"{self.dataSet.samples[i][0]},cls:{self.dataSet.samples[i][1]}")
        self.imageCanvas.showImage(self.img)
        if self.rrcbtn.isChecked():
            x = RRCTensor(self.img)
        else:
            x = transToTensor(self.img)
        x = transToStd(x)
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
        self.models = {
            "None": lambda: None,
            "vgg16": lambda: tv.models.vgg16(weights=tv.models.VGG16_Weights.DEFAULT),
            # "alexnet": lambda: tv.models.alexnet(weights=tv.models.AlexNet_Weights.DEFAULT),
            "alexnet": lambda: None, # not support for lrp
        }
        self.model = None

        # partial fun 参数是静态的，传了就不能变，此处要求每次访问self.model。（写下语句的时候就创建完了）
        # lambda fun 是动态的，运行时解析
        # 结合一下匿名lambda函数就可以实现 创建含动态参数(model)的partial fun，只多了一步调用()
        self.methods = {
            "None":         lambda: None,
            "GradCAM":      lambda: partial(GradCAM({'type': 'vgg16',
                                                     'arch': self.model,
                                                     'layer_name': '30',
                                                     'input_size': (224, 224)}
                                                    ).__call__,
                                            sg=False,relu=False),
            "SG GradCAM":   lambda: partial(GradCAM({'type': 'vgg16','arch': self.model,
                                             'layer_name': '30','input_size': (224, 224)}).__call__,
                                            sg=True,relu=False),
            "LayerCAM":     lambda :partial(LayerCAM({'type': 'vgg16','arch': self.model,
                                             'layer_name': '30','input_size': (224, 224)}).__call__,
                                            sg=False,relu=False),
            "SG LayerCAM":  lambda: partial(LayerCAM({'type': 'vgg16', 'arch': self.model,
                                                  'layer_name': '30', 'input_size': (224, 224)}).__call__,
                                            sg=True, relu=False),
            "LRP":          lambda: lambda x, y: LRP_Generator(self.model)(x, y,backward_init='origin', method='lrpc')[0].sum(1, True),
            "SGLRP":        lambda: lambda x, y: LRP_Generator(self.model)(x, y,backward_init='sglrp', method='lrpc')[0].sum(1, True),
            "LRP ZP ":      lambda: lambda x, y: LRP_Generator(self.model)(x, y,backward_init='origin', method='lrpzp')[0].sum(1, True),
            "LRP ZP SG":    lambda: lambda x, y: LRP_Generator(self.model)(x, y,backward_init='sglrp', method='lrpzp')[0].sum(1, True),
        }
        self.method = None
        self.img = None

        self.masks = {
            "Raw Heatmap":      lambda hm,im:(None,             normalize_R(hm)),
            "Overlap":          lambda hm,im:(InverseStd(im),   normalize_R(hm)),
            "Positive Only":    lambda hm,im:(InverseStd(im * positize(hm)), None),
            "Sparsity 50":      lambda hm,im:(InverseStd(im * binarize(hm, sparsity=0.5)), None),
            "Maximal Patch":    lambda hm,im:(InverseStd(im * maximalPatch(hm, top=True, r=10)), None),
        }
        self.mask = None

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        self.setMaximumWidth(600)
        self.setTitle("Explain Method Selector")

        hlayout = QHBoxLayout()
        self.modelSelect = QComboBox()
        temp = QStandardItemModel()
        for key in self.models:
            temp2 = QStandardItem(key)
            temp2.setData(key)
            temp2.setSizeHint(QSize(200, 40))
            temp.appendRow(temp2)
        self.modelSelect.setModel(temp)
        self.modelSelect.setCurrentIndex(1)
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
        self.methodSelect.setCurrentIndex(1)
        self.methodSelect.setMinimumHeight(40)
        hlayout.addWidget(TipedWidget("Method: ", self.methodSelect))
        main_layout.addLayout(hlayout)
        del hlayout

        hlayout=QHBoxLayout()
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
        self.classSelector.setMaximumWidth(500)
        self.classSelector.setPlaceholderText("classes choose:")
        main_layout.addWidget(TipedWidget("Classes:", self.classSelector))




        # class sort
        self.topk = 15
        main_layout.addWidget(QLabel(f"Prediction Top {self.topk}"))
        self.predictionScreen = QPlainTextEdit("this is place classes predicted shown\nexample:class 1 , Cat")
        self.predictionScreen.setMinimumHeight(40)
        self.predictionScreen.setMaximumWidth(800)
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
        self.ImageChange()

    def methodChange(self):
        if self.model is None:
            return
        t = self.methodSelect.currentText()
        self.method = self.methods[t]()
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
        self.classes = imgnt.classes

    def ImageChange(self, x=None):
        if x is not None:
            self.img = x
        if self.img is None:
            return
        # 测试，输出传入的图像
        self.imageCanvas.showImage(ToPlot(InverseStd(self.img)))
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
                "\n".join(f"{i}:\t{v:.4f}:\t{self.classes[i]}" for i, v in zip(topki, topkv))
            )
            self.classSelector.setText(",".join(str(i.item()) for i in topki))
            if self.method is not None:
                self.HeatMapChange()

    def HeatMapChange(self):
        if self.img is None or self.model is None or self.method is None:
            return
        try:
            # get classes
            classes = list(int(cls) for cls in self.classSelector.text().split(','))
            # classes=classes[:self.maxHeatmap] # always 6
            classes = classes[:6]
        except Exception as e:
            self.predictionScreen.setPlainText(e.__str__())
            return
        # heatmaps
        # self.imageCanvas.clear()
        # 必须用对象自带的figure，否则大小不能自动调整
        fig = self.imageCanvas.figure
        fig.clf()
        # 显存有时候会超限
        # tc.cuda.empty_cache()
        fig.tight_layout()
        # fig.subplots(3,2)
        # 强制创建六张图
        sfs = fig.subfigures(3, 2)
        # 当然每张图分为左右部分，左边是热力图，右边是例子
        # hmpair=[]
        # ___________runningCost___________ = RunningCost(50)
        # ___________runningCost___________.tic()
        for i, cls in enumerate(classes):
            sf = sfs.flatten()[i]
            # sf.set_title(str(cls))
            # sf.tight_layout()
            # axe=fig.add_subplot(3,2,i+1)
            hm = self.method(self.img_dv, cls).cpu().detach()
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
            if self.mask is not None:
                img,ovlap=self.mask(hm,self.img)
                if img is not None:
                    axe.imshow(ToPlot(img), cmap=None, vmin=0, vmax=1)
                    if ovlap is not None:
                        axe.imshow(ToPlot(ovlap), cmap=lrp_cmap, alpha=0.7, vmin=-1, vmax=1)
                elif ovlap is not None:
                    axe.imshow(ToPlot(ovlap), cmap=lrp_cmap, vmin=-1, vmax=1)
            # else:
            #     axe.imshow(format_for_plotting(normalize_R(hm)), cmap=lrp_cmap, vmin=-1, vmax=1)
            axe.set_axis_off()
            axe.set_title(str(cls))
            axe = sf.add_subplot(1, 2, 2)
            axe.imshow(example)
            axe.set_axis_off()
            # pair=[hm,example]
            # hmpair.append(pair)
            # ___________runningCost___________.tic("ploting")
            # time of plotting chasing time of generating heatmap. any alternative?
        # ___________runningCost___________.cost()

        self.imageCanvas.showFigure(fig)

        # for hm,ex in hmpair:
        #     pass

        # self.imageCanvas.showAxe(axe)

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
        self.showMaximized()


def main():
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
