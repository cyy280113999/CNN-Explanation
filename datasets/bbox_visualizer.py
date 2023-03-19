# sys
import os
import sys
import random
from functools import partial
# nn
import numpy as np
import torch as tc
import torchvision.transforms.functional as ttf

# gui
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QPainter
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QGroupBox, QVBoxLayout, QComboBox, QPushButton, QLineEdit, \
    QFileDialog, QMainWindow, QApplication
# draw
from PIL import Image

import pyqtgraph as pg

pg.setConfigOptions(**{'imageAxisOrder': 'row-major',
                       # 'useNumba': True,
                       # 'useCupy': True,
                       })


# user
from utils import *
from bbox_imgnt import BBImgnt


# torch initial
device = "cuda"

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

class ImageLoader(QGroupBox):
    def __init__(self):
        super().__init__()
        # key: dataset name , value: is folder or not
        self.classes = loadImageNetClasses()
        self.dataSets = {
            "ImageNet Val": lambda: BBImgnt(),
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

        self.back = QPushButton("Back")
        self.next = QPushButton("Next")
        self.index = QLineEdit("0")

        hlayout.addWidget(self.back)
        hlayout.addWidget(self.next)
        hlayout.addWidget(self.index)
        main_layout.addLayout(hlayout)

        self.back.setMinimumHeight(40)
        self.next.setMinimumHeight(40)
        self.index.setMinimumHeight(40)
        self.index.setMaximumWidth(80)
        self.index.setMaxLength(8)

        # image information
        self.imgInfo = QLabel("Image")
        main_layout.addWidget(self.imgInfo)

        # image show
        self.imageCanvas = ImageCanvas()
        # self.imageCanvas.showImage(loadTestImg())
        main_layout.addWidget(self.imageCanvas)

        self.dataSetSelect.currentIndexChanged.connect(self.dataSetChange)
        self.next.clicked.connect(self.indexNext)
        self.back.clicked.connect(self.indexBack)
        self.index.returnPressed.connect(self.imageChange)

        self.dataSetChange()


    def dataSetChange(self):
        t = self.dataSetSelect.currentText()
        if t == "Customized Folder":
            []
        else:
            self.dataSet = self.dataSets[t]()
            self.next.show()
            self.back.show()
            self.index.show()
            self.dataSetLen.setText(f"Images Index From 0 to {len(self.dataSet) - 1}")
            self.index.setText("0")
            self.imageChange()

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
        i = self.checkIndex()
        img, cls, bboxs = self.dataSet[i]
        img = toPlot(invStd(img)).numpy()
        self.img = img
        self.imgInfo.setText(f"bbox:{bboxs[0]}cls:{cls}")
        # cmap=pcolors.get()
        self.imageCanvas.pglw.clear()
        pi: pg.PlotItem = self.imageCanvas.pglw.addPlot()
        im = pg.ImageItem(img, autolevel=False, autorange=False)  # ,levels=[0,1])#,lut=None)
        pi.addItem(im)
        for bbox in bboxs:
            xmin, ymin, xmax, ymax=bbox
            boxpdi = pg.PlotDataItem(x=[xmin, xmax, xmax, xmin, xmin], y=[ymin, ymin, ymax, ymax, ymin])
            pi.addItem(boxpdi)
        pi.showAxes(True)
        pi.invertY(True)
        pi.vb.setAspectLocked(True)

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




class ImageCanvas(QGroupBox):
    def __init__(self):
        super(ImageCanvas, self).__init__()
        self.setTitle("Image:")
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # self.imv = pg.ImageView()
        # self.imv.setImage()
        self.pglw = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.pglw)

    def showImage(self, img):
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


    def showImages(self, imgs):
        pass

    def clear(self):
        self.imv.clear()


class MainWindow(QMainWindow):
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

        # 左屏幕
        self.imageLoader = ImageLoader()
        cleft_panel.addWidget(self.imageLoader)


        self.show()


def main():
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
