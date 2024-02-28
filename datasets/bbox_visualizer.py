# sys

# nn
import numpy as np

# gui
from PyQt5.QtWidgets import QWidget, QHBoxLayout

# draw

# user
from utils.plot import pg
from utils.window_tools import BaseDatasetTravellingVisualizer, windowMain
from pyqtgraph import mkPen
from bbox_imgnt import BBImgnt

# torch initial
device = "cuda"


class BBoxImgntTravellingVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ImageNet BBox Visualizer')
        self.main_layout=QHBoxLayout()
        self.setLayout(self.main_layout)
        self.dataSet = BBImgnt(transform=None)
        # use this trick instead of inheriting class
        self.canvas=BaseDatasetTravellingVisualizer(self.dataSet, imageChangeCallBack=self.imageChange)
        self.main_layout.addWidget(self.canvas)

    def imageChange(self, vser):
        # when member function c.F created by c=C() called, it pushes self as it first parameter
        # so c.F() is like C.F(c)
        img, cls, bboxs = vser.raw_inputs
        img = np.array(img)
        vser.img_info.setText(f"bbox:{bboxs[0]}cls:{cls}")
        vser.imageCanvas.pglw.clear()
        # manually adding drawing items
        pi: pg.PlotItem = vser.imageCanvas.pglw.addPlot()
        im = pg.ImageItem(img, autolevel=False, autorange=False)  # ,levels=[0,1])#,lut=None)
        pi.addItem(im)
        for bbox in bboxs:
            xmin, ymin, xmax, ymax = bbox
            boxpdi = pg.PlotDataItem(x=[xmin, xmax, xmax, xmin, xmin],
                                     y=[ymin, ymin, ymax, ymax, ymin], pen=mkPen(color='g', width=3))
            pi.addItem(boxpdi)
        pi.showAxes(True)
        pi.invertY(True)
        pi.vb.setAspectLocked(True)


if __name__ == "__main__":
    windowMain(BBoxImgntTravellingVisualizer)
