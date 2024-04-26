import numpy as np
import sys

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, \
    QComboBox, QPushButton, QAbstractItemView, QFileDialog
import pyqtgraph as pg
from utils import pyqtgraphDefaultConfig
pyqtgraphDefaultConfig()
from pyqtgraph.functions import mkPen
from config import select_name, select_flag

class ColorTool:
    def __init__(self, color_count):
        self.colors = {}
        for i in range(color_count):
            self.colors[i] = pg.intColor(i, color_count)


class PenTool:
    colors = [
              QColor(50, 50, 50),   # 黑色
              QColor(255, 0, 0),    # 红色
              QColor(0, 150, 0),    # 绿色
              QColor(255, 0, 255),  # 紫色
              QColor(165, 42, 42),  # 褐色
              QColor(0, 255, 200),  # 青色
              QColor(255, 165, 0),  # 橙色
              QColor(128, 255, 128),# 浅绿
              QColor(255, 192, 203),# 粉色
              QColor(0, 0, 255),    # 蓝色
              ]
    def __init__(self, color_count, pen_width=3):
        self.pens = []
        if color_count>len(self.colors):
            color_tool=lambda x:pg.intColor(x,color_count)
        else:
            self.colors=self.colors[1:color_count]+[self.colors[0]]  # last is black
            color_tool=lambda x:self.colors[x]
        for i in range(color_count):
            self.pens.append(mkPen(color=color_tool(i), width=pen_width))


class BaseAnalyser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.init_UI()

    def init_UI(self):
        self.hlayout = QHBoxLayout()
        self.main_widget.setLayout(self.hlayout)

        self.control_widget = QWidget()
        self.hlayout.addWidget(self.control_widget)

        self.vlayout = QVBoxLayout()
        self.control_widget.setLayout(self.vlayout)

        # create load and analyse button
        self.select_file_btn = QPushButton('Customize Log File')
        self.analyze_btn = QPushButton('Analyse')

        # create dataset filters
        self.dataset_label = QLabel('Dataset:')
        self.dataset_combo = QComboBox()

        # create model filters
        self.model_label = QLabel('Model:')
        self.model_combo = QComboBox()

        # create method filters
        self.method_label = QLabel('Methods:')
        self.method_list = QListWidget()
        self.method_list.setSelectionMode(QAbstractItemView.MultiSelection)

        self.vlayout.addWidget(self.select_file_btn)
        self.vlayout.addWidget(self.analyze_btn)
        self.vlayout.addWidget(self.dataset_label)
        self.vlayout.addWidget(self.dataset_combo)
        self.vlayout.addWidget(self.model_label)
        self.vlayout.addWidget(self.model_combo)
        self.vlayout.addWidget(self.method_label)
        self.vlayout.addWidget(self.method_list)
        self.control_widget.setMaximumWidth(400)

        # function signals
        self.select_file_btn.clicked.connect(self.select_file)
        self.analyze_btn.clicked.connect(self.load_data)
        self.dataset_combo.currentIndexChanged.connect(self.dataset_change)
        self.model_combo.currentIndexChanged.connect(self.model_change)
        self.method_list.itemSelectionChanged.connect(self.method_change)

        self.customized_UI()

    def select_file(self):
        filename_long, f_type = QFileDialog.getOpenFileName(directory="../datas/")
        if filename_long:
            self.filename=filename_long

    def load_data(self):
        # Read data from file
        with open(self.filename, 'r') as f:
            data = f.readlines()
        # remove empty line
        data = [l.strip('\n') for l in data]
        data = [l.split(',') for l in data if len(l)>0]
        # Convert data to numpy array
        self.data = np.array(data)
        # Get unique dataset names
        self.ds_names = np.unique(self.data[:, 0])
        self.dataset_combo.clear()
        self.dataset_combo.addItems(self.ds_names)
        self.dataset_change()

    def dataset_change(self):
        self.ds_name=self.dataset_combo.currentText()
        self.ds_data=self.data[self.data[:,0]==self.ds_name]
        self.model_names = np.unique(self.ds_data[:, 1]).tolist()
        self.model_names.sort()
        self.model_combo.clear()
        self.model_combo.addItems(self.model_names)
        self.model_change()

    def model_change(self):
        self.model_name=self.model_combo.currentText()
        self.model_data=self.ds_data[self.ds_data[:, 1] == self.model_name]
        self.method_names = np.unique(self.model_data[:, 2]).tolist()
        self.method_names.sort()
        self.method_list.clear()
        self.method_list.addItems(self.method_names)
        self.method_list.setCurrentRow(0)
        self.method_change()

    def method_change(self):
        method_filter = [item.text() for item in self.method_list.selectedItems()]
        self.method_data=self.model_data
        if method_filter:
            self.method_data = [row for row in self.method_data if row[2] in method_filter]
            self.method_data.sort(key=lambda x:x[2])
            self.method_data = np.vstack(self.method_data)
        self.current_data = self.method_data
        self.paperSort()  # usually turn off
        self.plot_data()

    def customized_UI(self):
        self.setWindowTitle('Base Analyser')
        self.filename = '../datas/log.txt'
        raise NotImplementedError()

    def plot_data(self):
        raise NotImplementedError()

    def paperSort(self):
        if not select_flag:
            return
        i=np.argwhere(self.current_data[:,2]==select_name)
        if len(i)!=0:
            i=i[0].item()
            self.current_data=np.concatenate([self.current_data[:i],self.current_data[i+1:],self.current_data[i,None]],axis=0)


def analyse_main(AnalyserClass:type):
    app = QApplication(sys.argv)
    ex = AnalyserClass()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    analyse_main(BaseAnalyser)