# sys
import os
import sys
import random
from functools import partial
from math import ceil
# nn
import numpy as np
import torch
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

# draw

# torch initial
device = "cuda"
torch.set_grad_enabled(False)

from datasets.OnlyImages import OnlyImages
from datasets.DiscrimDataset import DiscrimDataset

imageNetVal = getImageNet('val', None)


class DataSetLoader(QGroupBox):
    def __init__(self, dataset=None):
        super().__init__()
        # key: dataset name , value: is folder or not
        self.classes = loadImageNetClasses()
        self.available_datasets = AvailableMethods({  # attr : name
            "CusImage": "Customized Image",
            "CusFolder": "Customized Folder",
            "ImgVal": "ImageNet Val",
            "ImgTrain": "ImageNet Train",
            "Discrim": "Discrim DataSet",
        })
        self.dataSets = {
            self.available_datasets.CusImage: None,
            self.available_datasets.CusFolder: None,
            self.available_datasets.ImgVal: lambda: imageNetVal,
            self.available_datasets.ImgTrain: lambda: getImageNet('train', None),
            self.available_datasets.Discrim: lambda: DiscrimDataset(transform=None, MultiLabel=False),
        }
        self.dataSet = None
        self.img = None
        # self.setMaximumWidth(600)
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.setTitle("Image Loader")

        # data set
        hlayout = QHBoxLayout()
        self.dataSetSelect = DictCombleBox(self.dataSets)
        # self.dataSetSelect.resize(200,40)
        self.open = QPushButton("Open")
        # self.open.setFixedSize(80,40)
        self.open.setMinimumHeight(40)
        hlayout.addWidget(QLabel("Data Set: "))
        hlayout.addWidget(self.dataSetSelect)
        hlayout.addWidget(self.open)
        self.main_layout.addLayout(hlayout)

        # data set info
        self.dataInfo = QLabel("dataset info:")
        self.main_layout.addWidget(self.dataInfo)

        # image choose
        hlayout = QHBoxLayout()
        self.backbtn = QPushButton("Back")
        self.nextbtn = QPushButton("Next")
        self.randbtn = QPushButton("Rand")
        self.indexEdit = QLineEdit("0")

        hlayout.addWidget(self.backbtn)
        hlayout.addWidget(self.nextbtn)
        hlayout.addWidget(self.randbtn)
        hlayout.addWidget(self.indexEdit)
        self.main_layout.addLayout(hlayout)


        # self.back.setFixedSize(80,40)
        # self.next.setFixedSize(80,40)
        self.backbtn.setMinimumHeight(40)
        self.nextbtn.setMinimumHeight(40)
        self.randbtn.setMinimumHeight(40)
        self.indexEdit.setMinimumHeight(40)
        self.indexEdit.setMaximumWidth(80)
        self.indexEdit.setMaxLength(8)

        # image information
        self.imgInfo = QLabel("Image")
        self.main_layout.addWidget(self.imgInfo)

        # new line
        hlayout = QHBoxLayout()
        # rrc switch
        self.rrcbtn = QPushButton("RRC")
        self.rrcbtn.setMinimumHeight(40)
        self.rrcbtn.setCheckable(True)
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
        self.modeSelects = DictCombleBox(self.modes)
        hlayout.addWidget(QLabel('Image modify:'))
        hlayout.addWidget(self.rrcbtn)
        hlayout.addWidget(self.modeSelects)

        # re-
        self.regeneratebtn = QPushButton("ReGenerate")
        self.regeneratebtn.setMinimumHeight(40)
        hlayout.addWidget(self.regeneratebtn)
        self.main_layout.addLayout(hlayout)

        # image show
        self.imageCanvas = ImageCanvas()
        # self.imageCanvas.showImage(loadTestImg())
        self.main_layout.addWidget(self.imageCanvas)

        # actions
        # def refresh(self,x=None):
        #
        # self.dataSetLen.refresh=refresh
        self.dataSetSelect.currentIndexChanged.connect(self.dataSetChange)
        self.open.clicked.connect(self.openSelect)
        self.nextbtn.clicked.connect(self.indexNext)
        self.backbtn.clicked.connect(self.indexBack)
        self.randbtn.clicked.connect(self.indexRand)
        self.indexEdit.returnPressed.connect(self.imageChange)

        # self.rrcbtn.clicked.connect(lambda :self.rrcbtn.setChecked(not self.rrcbtn.isChecked()))
        self.modeSelects.currentIndexChanged.connect(self.modeChange)
        self.regeneratebtn.clicked.connect(self.imageChange)

    def dataSetChange(self):
        t = self.dataSetSelect.currentText()
        if t == self.available_datasets.CusFolder:
            self.open.setEnabled(True)
            self.nextbtn.show()
            self.backbtn.show()
            self.randbtn.show()
            self.indexEdit.show()
            self.dataInfo.setText(f"Please select folder")
        elif t == self.available_datasets.CusImage:
            self.open.setEnabled(True)
            self.nextbtn.hide()
            self.backbtn.hide()
            self.randbtn.hide()
            self.indexEdit.hide()
            self.dataInfo.setText(f"Please open image")
        else:
            self.dataSet = self.dataSets[t]()
            self.open.setEnabled(False)
            self.nextbtn.show()
            self.backbtn.show()
            self.randbtn.show()
            self.indexEdit.show()
            self.dataInfo.setText(f"Images Index From 0 to {len(self.dataSet) - 1}")
            self.checkIndex(0)
            self.imageChange()

    def openSelect(self):
        t = self.dataSetSelect.currentText()
        if t == self.available_datasets.CusFolder:
            directory = QFileDialog.getExistingDirectory(directory="./")
            if directory:
                subdir = [entry for entry in os.scandir(directory) if entry.is_dir()]
                if not subdir:
                    self.dataSet = OnlyImages(directory)
                else:
                    self.dataSet = tv.datasets.ImageFolder(directory)
                self.dataInfo.setText(f"Images Index From 0 to {len(self.dataSet) - 1}")
                self.indexEdit.setText("0")
                self.imageChange()

        elif t == self.available_datasets.CusImage:
            filename_long, f_type = QFileDialog.getOpenFileName(directory="./")
            if filename_long:
                self.img = pilOpen(filename_long)
                # self.img = np.asarray(img_PIL)
                self.imgInfo.setText(filename_long)
                self.imageCanvas.showImage(np.array(self.img))
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
            self.indexEdit.setText(str(i))
        else:
            i = int(self.indexEdit.text())
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
        if t == self.available_datasets.CusFolder:
            if self.dataSet is None:
                return
            i = self.checkIndex()
            self.img, _ = self.dataSet[i]
            path, cls = self.dataSet.samples[i]
            self.imgInfo.setText(f"{path},cls:{cls}")
        elif t == self.available_datasets.CusImage:
            pass
        elif t == self.available_datasets.Discrim:
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
            self.img_t = toTensorS224(self.img)
            if self.rrcbtn.isChecked():
                self.img_t = toRRC(self.img_t)
            self.img_t = toStd(self.img_t)
            if self.imageMode:
                self.img_t = self.imageMode(self.img_t)
            self.imageCanvas.showImage(toPlot(invStd(self.img_t)).clip(min=0, max=1))
            self.img_t = self.img_t.unsqueeze(0)
            self.sendImg(self.img_t)

    def sendImg(self, x):
        if hasattr(self, 'emitter') and self.emitter is not None:
            self.emitter.emit(self.img_t)

    def init(self, send_signal: pyqtSignal):
        self.emitter = send_signal
        self.dataSetChange()


class ExplainMethodSelector(QGroupBox):
    def __init__(self):
        super(ExplainMethodSelector, self).__init__()

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        # self.setMaximumWidth(600)
        self.setTitle("Explain Method Selector")

        self.models = {
            # "None": lambda: None,
            k: v for k, v in avaiable_models.items()
        }
        self.model = None

        from HeatmapMethods import heatmap_methods
        self.methods = heatmap_methods

        # the mask interface, all masks must follow this:
        # the masked heatmap generated by calling "im, cover = m(hm, im)"
        # the param hm is raw heatmap, the im is input image
        # the output im is a valid printable masked heatmap image
        # the output hm is raw heatmap
        # mask: hm, im -> image[0,1], covered heatmap[-1,1]
        self.avaiable_masks=AvailableMethods({
            "RawH": "Raw Heatmap",
            "Overlap": "Overlap",
        })
        self.masks = {
            "Raw Heatmap": lambda hm, im: (None, hm),
            "Overlap": lambda hm, im: (invStd(im), hm),
            "Reversed Image": lambda hm, im: (hm, None),
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
        self.modelSelect = DictCombleBox(self.models)
        hlayout.addWidget(TippedWidget("Model: ", self.modelSelect))

        self.methodSelect = DictCombleBox(self.methods)
        hlayout.addWidget(TippedWidget("Method: ", self.methodSelect))
        self.main_layout.addLayout(hlayout)
        del hlayout

        hlayout = QHBoxLayout()
        self.maskSelect = DictCombleBox(self.masks)
        hlayout.addWidget(TippedWidget("Mask: ", self.maskSelect))

        self.regeneratebtn1 = QPushButton("Reload Image")
        self.regeneratebtn1.setMinimumHeight(40)
        hlayout.addWidget(self.regeneratebtn1)
        # re-
        self.regeneratebtn2 = QPushButton("ReGenerate Heatmap")
        self.regeneratebtn2.setMinimumHeight(40)
        hlayout.addWidget(self.regeneratebtn2)
        self.main_layout.addLayout(hlayout)
        del hlayout

        # class
        self.classSelector = QLineEdit("")
        self.classSelector.setMinimumHeight(40)
        # self.classSelector.setMaximumWidth(500)
        self.classSelector.setPlaceholderText("classes choose:")
        self.main_layout.addWidget(TippedWidget("Classes:", self.classSelector))

        # class sort
        self.topk = 10
        self.main_layout.addWidget(QLabel(f"Prediction Top {self.topk}"))
        self.predictionScreen = QPlainTextEdit("this is place classes predicted shown\nexample:class 1 , Cat")
        self.predictionScreen.setMinimumHeight(40)
        # self.predictionScreen.setMaximumWidth(800)
        self.predictionScreen.setReadOnly(True)
        self.main_layout.addWidget(self.predictionScreen)

        # actions
        self.modelSelect.currentIndexChanged.connect(self.modelChange)
        self.methodSelect.currentIndexChanged.connect(self.methodChange)
        self.maskSelect.currentIndexChanged.connect(self.maskChange)

        self.regeneratebtn1.clicked.connect(self.callImgChg)
        self.classSelector.returnPressed.connect(self.heatmapChange)
        self.regeneratebtn2.clicked.connect(self.heatmapChange)

    def init(self, receive_signal, imgnt, canvas=None):
        # output canvas
        self.maxHeatmap = 6
        if canvas is not None:
            self.imageCanvas = canvas
        else:
            self.imageCanvas = ImageCanvas()  # no add
        self.imgnt = imgnt
        self.classes = loadImageNetClasses()
        self.reciever = receive_signal
        self.reciever.connect(self.imageChange)
        self.modelChange()
        self.maskChange()

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
                self.heatmapChange()

    def maskChange(self):
        t = self.maskSelect.currentText()
        self.mask = self.masks[t]
        self.heatmapChange()

    def imageChange(self, x=None):
        if x is not None:
            self.img = x
        if self.img is None:
            return
        # test, send img to canvas
        # self.imageCanvas.showImage(ToPlot(InverseStd(self.img)))
        if self.model is not None:
            # predict
            self.img_dv = self.img.to(device)
            prob = torch.softmax(self.model(self.img_dv), 1)
            topk = prob.sort(1, True)
            topki = topk[1][0, :self.topk]
            topkv = topk[0][0, :self.topk]
            # show info
            if self.classes is None:
                raise Exception("saveClasses First.")
            self.predictionScreen.setPlainText(
                "\n".join(self.PredInfo(i.item(), v.item(), 50) for i, v in zip(topki, topkv))
            )
            self.classSelector.setText(",".join(str(i.item()) for i in topki))
            self.heatmapChange()

    def PredInfo(self, cls, prob=None, max_len=30):
        if prob:
            s = f"{cls}:\t{prob:.4f}:\t{self.classes[cls]}"
        else:
            s = f"{cls}:{self.classes[cls]}"
        return s + (max_len-len(s))*" "

    def heatmapChange(self):
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
        hms=[]
        with torch.enable_grad():
            for cls in classes:
                hms.append(self.method(self.img_dv, cls).detach().cpu())
        self.imageCanvas.pglw.clear()
        # self.imageCanvas.showImages(hms)
        # ___________runningCost___________ = RunningCost(20)
        row_count = 2
        for i, cls in enumerate(classes):
            # ___________runningCost___________.tic()
            row = i / row_count
            col = i % row_count
            # self.imageCanvas.pglw.nextRow()
            l = self.imageCanvas.pglw.addLayout(row=row, col=col)  # 2 images
            hm=hms[i]
            # ___________runningCost___________.tic("generate heatmap")
            example = self.cls_example(cls)
            im_exp = toPlot(toTensorS224(example))
            # ___________runningCost___________.tic("prepare example")
            pi: pg.PlotItem = l.addPlot(0, 0)
            plotItemDefaultConfig(pi)
            p = None
            if self.mask is not None:
                masked, covering = self.mask(hm, self.img)
                if masked is not None:
                    p = self.maskScore(masked, cls)
                    pi.addItem(pg.ImageItem(toPlot(masked), levels=[0, 1], opacity=1.))
                if covering is not None:
                    pi.addItem(pg.ImageItem(toPlot(covering),
                                            # compositionMode=QPainter.CompositionMode.CompositionMode_Overlay,
                                            levels=[-1, 1], lut=lrp_lut, opacity=0.7))
            else:
                pi.addItem(pg.ImageItem(toPlot(hm), levels=[-1, 1], lut=lrp_lut))
            pi.setTitle(self.PredInfo(cls, p, 40))
            pexp = l.addPlot(0, 1)
            plotItemDefaultConfig(pexp)
            # hw=min(im_exp.shape[0],im_exp.shape[1])
            # pexp.setFixedWidth(500)
            # pexp.setFixedHeight(500)
            pexp.addItem(pg.ImageItem(im_exp))
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
        self.imageChange()


class MainWindow(QMainWindow):
    # must define signal in class
    imageChangeSignal = pyqtSignal(torch.Tensor)

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


def expVisMain(SeperatedWindow=True):
    global imageNetVal
    # --workflow
    # create window
    qapp = QApplication.instance()
    if qapp is None:
        qapp = QApplication(sys.argv)
    imageLoader = DataSetLoader()  # keep this instance alive!
    explainMethodsSelector = ExplainMethodSelector()
    imageCanvas = ImageCanvas()
    mw = MainWindow(imageLoader, explainMethodsSelector, imageCanvas, SeperatedCanvas=SeperatedWindow)
    # initial settings
    explainMethodsSelector.init(mw.imageChangeSignal, imageNetVal, canvas=imageCanvas)
    imageLoader.init(mw.imageChangeSignal)
    mw.show()
    sys.exit(qapp.exec_())


if __name__ == "__main__":
    expVisMain()
