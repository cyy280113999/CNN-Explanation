import sys
import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout, QPushButton, QLineEdit, QApplication, QComboBox
import pyqtgraph as pg

from .image_dataset import DatasetTraveller
from .hm_plot import plotItemDefaultConfig, lrp_lut


class TippedWidget(QWidget):
    def __init__(self, tip="Empty Tip", widget=None):
        super(TippedWidget, self).__init__()
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        main_layout.addWidget(QLabel(tip))
        if widget is None:
            raise Exception("Must given widget.")
        self.widget = widget
        main_layout.addWidget(self.widget)

    def __getitem__(self, item):
        return self.widget.__getitem__(item)


class DictCombleBox(QComboBox):
    def __init__(self, combo_dict, ShapeMode=1):
        super().__init__()
        if ShapeMode == 0:
            for k, v in combo_dict.items():
                self.addItem(k)
        elif ShapeMode == 1:
            temp = QStandardItemModel()
            for key in combo_dict:
                temp2 = QStandardItem(key)
                temp2.setData(key)  # , Qt.ToolTipRole
                temp2.setSizeHint(QSize(200, 40))
                temp.appendRow(temp2)
            self.setModel(temp)
        self.setCurrentIndex(0)
        self.setMinimumHeight(40)


class ImageCanvas(QWidget):
    def __init__(self):
        super(ImageCanvas, self).__init__()
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        self.pglw: pg.GraphicsLayout = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.pglw)

    def showImage(self, img, levels=(0, 1), lut=None):
        assert isinstance(img, np.ndarray)
        self.pglw.clear()
        pi: pg.PlotItem = self.pglw.addPlot()
        plotItemDefaultConfig(pi)
        ii = pg.ImageItem(img, levels=levels, lut=lut)
        pi.addItem(ii)

    def showImages(self, imgs, size=(1, 1), levels=(0, 1), lut=None):
        for i, img in enumerate(imgs):
            row = i // size[0]
            col = i % size[0]
            pi: pg.PlotItem = self.pglw.addPlot(row=row, col=col)
            plotItemDefaultConfig(pi)
            ii = pg.ImageItem(img, levels=levels, lut=lut)
            pi.addItem(ii)


class BaseDatasetTravellingVisualizer(QWidget):
    """
    if it is not inherited,  use imageChangeCallBack=your_call_back instead.
    """
    def __init__(self, dataset, AddCanvas=True, imageChangeCallBack=None):
        super().__init__()
        self.dataSet = dataset
        self.imageSelector = DatasetTraveller(self.dataSet)
        self.raw_inputs = None
        self.initUI()
        # canvas
        if AddCanvas:
            self.imageCanvas = ImageCanvas()
            self.main_layout.addWidget(self.imageCanvas)
        if imageChangeCallBack is not None:
            self.imageChange = lambda: imageChangeCallBack(self)
        self.getImage()

    def initUI(self):
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        hlayout = QHBoxLayout()  # add row
        self.dataSetInfo = QLabel()
        self.dataSetInfo.setText(f"Images Index From 0 to {len(self.dataSet) - 1}")
        hlayout.addWidget(self.dataSetInfo)

        self.back = QPushButton("Back")
        self.next = QPushButton("Next")
        self.randbtn = QPushButton("Rand")
        self.index = QLineEdit("0")

        hlayout.addWidget(self.back)
        hlayout.addWidget(self.next)
        hlayout.addWidget(self.randbtn)
        hlayout.addWidget(self.index)
        self.main_layout.addLayout(hlayout)
        self.back.setMinimumHeight(40)
        self.next.setMinimumHeight(40)
        self.randbtn.setMinimumHeight(40)
        self.index.setMinimumHeight(40)
        self.index.setMaximumWidth(80)
        self.index.setMaxLength(8)

        self.imgInfo = QLabel("Image Info:")
        self.main_layout.addWidget(self.imgInfo)
        self.next.clicked.connect(self.indexNext)
        self.back.clicked.connect(self.indexBack)
        self.randbtn.clicked.connect(self.indexRand)
        self.index.returnPressed.connect(self.getImage)

    def indexNext(self):
        self.raw_inputs = self.imageSelector.next()
        self.index.setText(str(self.imageSelector.index))
        self.imageChange()

    def indexBack(self):
        self.raw_inputs = self.imageSelector.back()
        self.index.setText(str(self.imageSelector.index))
        self.imageChange()

    def indexRand(self):
        self.raw_inputs = self.imageSelector.rand()
        self.index.setText(str(self.imageSelector.index))
        self.imageChange()

    def getImage(self):
        self.raw_inputs = self.imageSelector.get(int(self.index.text()))
        self.index.setText(str(self.imageSelector.index))
        self.imageChange()

    def imageChange(self):
        # img, cls = self.raw_input
        # self.imageCanvas.showImage(np.array(img))
        raise NotImplementedError()


def windowMain(WindowClass):
    qapp = QApplication.instance()
    if qapp is None:
        qapp = QApplication(sys.argv)
    mw = WindowClass()
    mw.show()
    sys.exit(qapp.exec_())
