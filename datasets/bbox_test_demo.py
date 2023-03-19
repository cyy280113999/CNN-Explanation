from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import numpy as np
from utils import *
import pyqtgraph as pg
pyqtgraphDefaultConfig(pg)

# images_dir = '/kaggle/input/ship-detection/images/'
# annotations_dir = '/kaggle/input/ship-detection/annotations/'

def showBox():
    boxpi = pg.PlotDataItem(x=[1, 2, 2, 1, 1], y=[1, 1, 2, 2, 1])
    glw = pg.GraphicsLayoutWidget()
    p1 = glw.addPlot()
    p1.addItem(boxpi)
    glw.show()
    pg.exec()

def showBBox(xmin, ymin, xmax, ymax):
    boxpi = pg.PlotDataItem(x=[xmin, xmax, xmax, xmin, xmin], y=[ymin, ymin, ymax, ymax, ymin])
    glw = pg.GraphicsLayoutWidget()
    p1 = glw.addPlot()
    p1.addItem(boxpi)
    glw.show()
    pg.exec()

def readBBox():
    imgnt_dir = r'F:\DataSet\imagenet\\'
    img_dir=r'val\n01440764\\'
    ana_dir=r'ILSVRC2012_bbox_val_v3\val\\'
    img_name = r'ILSVRC2012_val_00000293'
    img_suffix = '.JPEG'
    ana_suffix = '.xml'

    filename=imgnt_dir+img_dir+img_name+img_suffix
    ananame =imgnt_dir+ana_dir+img_name+ana_suffix
    sample_image = Image.open(filename).convert('RGB')

    img=np.array(sample_image)
    glw = pg.GraphicsLayoutWidget()
    glw.show()
    # imv = pg.ImageView()
    # imv.setImage()
    # pg.image(img)
    imi=pg.ImageItem(img)
    imp=glw.addPlot()
    imp.addItem(imi)
    plotItemDefaultConfig(imp)

    tree = ET.parse(ananame)
    root = tree.getroot()
    sample_annotations = []
    for neighbor in root.iter('bndbox'):
        xmin = int(neighbor.find('xmin').text)
        ymin = int(neighbor.find('ymin').text)
        xmax = int(neighbor.find('xmax').text)
        ymax = int(neighbor.find('ymax').text)
        print(xmin, ymin, xmax, ymax)
        sample_annotations.append([xmin, ymin, xmax, ymax])
    sample_image_annotated = sample_image.copy()
    # draw in pil
    # img_bbox = ImageDraw.Draw(sample_image_annotated)
    # for bbox in sample_annotations:
    #     # print(bbox)
    #     img_bbox.rectangle(bbox, outline="green")
    img=np.array(sample_image_annotated)
    imi=pg.ImageItem(img)
    imp=glw.addPlot()
    imp.addItem(imi)
    plotItemDefaultConfig(imp)
    pg.exec()
readBBox()