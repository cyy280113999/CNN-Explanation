# sys

# nn
import numpy as np

# gui

# draw

# user
from utils.plot import pg
from utils.window_tools import DatasetTravellingVisualizer, windowMain
from pyqtgraph import mkPen
from bbox_imgnt import BBImgnt

# torch initial
device = "cuda"


class BBoxImgntTravellingVisualizer(DatasetTravellingVisualizer):
    def __init__(self):
        dataSet = BBImgnt(transform=None)
        super().__init__(dataSet)
        self.setWindowTitle('ImageNet BBox Visualizer')

    def imageChange(self):
        self.index.setText(str(self.imageSelector.index))
        img, cls, bboxs = self.raw_input
        img = np.array(img)
        self.imgInfo.setText(f"bbox:{bboxs[0]}cls:{cls}")
        self.imageCanvas.pglw.clear()
        # manually adding drawing items
        pi: pg.PlotItem = self.imageCanvas.pglw.addPlot()
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
