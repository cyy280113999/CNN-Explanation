import numpy as np
import sys

from PyQt5.QtCore import QItemSelectionModel
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, \
    QComboBox, QPushButton, QAbstractItemView, QFileDialog
import pyqtgraph as pg

class ColorTool:
    def __init__(self, count):
        self.count=count
        self.colors = {}
    def get_color(self, value):
        if value not in self.colors:
            self.colors[value] = pg.intColor(value, self.count)
        return self.colors[value]

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
        # Convert data to numpy array
        data = [l.strip('\n').split(',') for l in data]
        self.data = np.array(data)
        # Get unique dataset names
        self.ds_names = np.unique(self.data[:, 0])
        self.dataset_combo.clear()
        self.dataset_combo.addItems(self.ds_names)
        self.dataset_change()

    def dataset_change(self):
        self.ds_name=self.dataset_combo.currentText()
        self.ds_data=self.data[self.data[:,0]==self.ds_name]
        self.model_names = np.unique(self.ds_data[:, 1])
        self.model_combo.clear()
        self.model_combo.addItems(self.model_names)
        self.model_change()

    def model_change(self):
        self.model_name=self.model_combo.currentText()
        self.model_data=self.ds_data[self.ds_data[:, 1] == self.model_name]
        self.method_names = np.unique(self.ds_data[:, 2])
        self.method_list.clear()
        self.method_list.addItems(self.method_names)
        self.method_list.setCurrentRow(0)
        self.method_change()

    def method_change(self):
        method_filter = [item.text() for item in self.method_list.selectedItems()]
        method_filter.sort()
        self.method_data=self.model_data
        if method_filter:
            self.method_data = [row for row in self.method_data if row[2] in method_filter]
        self.current_data = self.method_data
        self.plot_data()

    def customized_UI(self):
        self.setWindowTitle('Base Analyser')
        self.filename = '../datas/log.txt'
        raise NotImplementedError()

    def plot_data(self):
        raise NotImplementedError()


def analyse_main(AnalyserClass:type):
    app = QApplication(sys.argv)
    ex = AnalyserClass()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    analyse_main(BaseAnalyser)