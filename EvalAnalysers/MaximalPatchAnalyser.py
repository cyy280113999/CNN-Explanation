import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QComboBox, QPushButton
import pyqtgraph as pg


# class MaximalPatchAnalyser1:
#     def __init__(self):
#
#         self.filename = '../datas/mplog.txt'
#
#         # colors_lib = [
#         #     'red', 'pink', 'yellow', 'orange', 'green', 'blue', 'purple', 'black'
#         # ]
#         self.colors_lib = list(mpl.colors.TABLEAU_COLORS.values())
#         # shuffle(colors_lib)
#
#
#     def analyze(self):
#         # 批量剪枝产生日志文件
#         # 本程序用来分析日志log.txt
#         # 包含filters remained（每层剩余个数），acc（准确度对比）
#
#
#         # rewrite by sorted method name, 重新按名称排序写入
#         SortByName = False
#
#         # methods filter
#         Filtering = True
#         #filters = ["SG-GradCAM-f", "SG-LRP-C-f", "ST-LRP-C-f", "SIG0-LRP-C-f"]
#         filters = 'GradCAM-f,LID-Taylor-sig-f,LRP-C-f,SG-GradCAM-f,SG-LRP-C-f,SG-LayerCAM-f,SIG0-LRP-C-f,ST-LRP-C-f,LID-IG-f,LID-Taylor-f'.split(',')
#         #filters = 'SIG0-LRP-C-1,IG,LID-IG-1,LID-IG-sig-1,LID-IG-sig-f,LID-Taylor-1,LID-Taylor-sig-1,ScoreCAM-f'.split(',')
#         with open(self.filename, 'r') as f:
#             data = f.readlines()
#
#         data = [l.strip('\n').split(',') for l in data]  # remove '\n'
#
#         if SortByName:
#             data.sort(key=lambda l: l[0])
#             with open(self.filename, 'w') as f:
#                 # 关于我忘记加回车不经过检查就写入然后改了半天这事
#                 # name_pt=r'[A-Z][a-zA-Z]*-[\w]*-?f'
#                 # result1 = re.sub(name_pt,lambda x:'\n'+x.group(),data)
#                 new_data = [','.join(l) + '\n' for l in data]
#                 f.writelines(new_data)
#
#         meta = ['dataset', 'model', "method", 'data']
#
#         data = np.array(data)
#
#         # split by experiment
#         ds = np.unique(data[:, 0])
#         assert len(ds)==1
#         md = np.unique(data[:, 1])
#         assert len(md)==1
#         methods = np.unique(data[:, 2])
#
#         if Filtering:
#             methods = [m for m in methods if m in filters]
#
#         colors = {i: j for i, j in zip(methods, self.colors_lib)}
#
#         fig = plt.figure()
#         axe1 = fig.add_subplot(1,2,1)
#         axe2 = fig.add_subplot(1,2,2)
#         for method in methods:
#             method_data = data[data[:, 2] == method]
#             assert len(method_data)==1
#             method_data=method_data[0]
#             maxacc = method_data[3:].astype(float)
#             minacc = maxacc[len(maxacc)//2:]
#             maxacc = maxacc[:len(maxacc)//2]
#             x=np.array([1, 2, 3, 5, 10, 20])
#
#             axe1.plot(x, maxacc, label=method, color=colors[method])
#             axe2.plot(x, minacc, label=method, color=colors[method])
#             axe1.set_xticks(x)
#             axe2.set_xticks(x)
#             axe1.grid('on')
#             axe2.grid('on')
#             # plt.xlim(0,5)
#             # plt.ylim(-0.001,0.0005)
#         axe1.legend()
#         axe2.legend()
#         axe1.set_title(f'max')
#         axe2.set_title(f'min')
#         plt.show()


# class MaximalPatchAnalyser2(QMainWindow):
#     def __init__(self):
#         # Initialize UI
#         self.initUI()
#
#     def initUI(self):
#         # Create main widget
#         self.main_widget = QWidget(self)
#
#         # Create data filtering controls
#         ds_label = QLabel('Dataset:')
#         self.ds_combo = QComboBox()
#         self.ds_combo.addItems(self.ds_names)
#         self.ds_combo.currentIndexChanged.connect(self.update_model_combo)
#
#         model_label = QLabel('Model:')
#         self.model_combo = QComboBox()
#         self.model_combo.currentIndexChanged.connect(self.update_analysis_list)
#
#         analysis_label = QLabel('Analysis:')
#         self.analysis_list = QListWidget()
#         self.analysis_list.setSelectionMode(QAbstractItemView.MultiSelection)
#         self.analysis_list.currentItemChanged.connect(self.update_plot)
#
#         # Create plot widgets
#         self.max_plot = pg.PlotWidget(title='Max')
#         self.min_plot = pg.PlotWidget(title='Min')
#
#
#         # Create layout and add widgets
#         layout = QGridLayout()
#         layout.addWidget(ds_label, 0, 0)
#         layout.addWidget(self.ds_combo, 0, 1)
#         layout.addWidget(model_label, 1, 0)
#         layout.addWidget(self.model_combo, 1, 1)
#         layout.addWidget(analysis_label, 2, 0)
#         layout.addWidget(self.analysis_list, 2, 1)
#         layout.addWidget(self.max_plot, 3, 0, 1, 2)
#         layout.addWidget(self.min_plot, 4, 0, 1, 2)
#
#         # Set main widget layout and add toolbar
#         self.main_widget.setLayout(layout)
#         self.setCentralWidget(self.main_widget)
#
#         # Set window properties
#         self.setWindowTitle('Maximal Patch Analyser')
#         self.setGeometry(100, 100, 800, 600)
#         self.show()
#
#     def load_data(self):
#         # Read data from file
#         with open(self.filename, 'r') as f:
#             data = f.readlines()
#
#         # Convert data to numpy array
#         data = [l.strip('\n').split(',') for l in data]
#         self.data = np.array(data)
#
#         # Get unique dataset names
#         self.ds_names = np.unique(self.data[:, 0])
#         self.model_names = np.unique(self.data[:, 1])
#         self.method_names = np.unique(self.data[:, 2])
#
#     def filter_data_by_ds(self, ds_name):
#         return self.data[self.data[:, 0] == ds_name]


from BaseAnalyser import *


class MaximalPatchAnalyser(BaseAnalyser):
    def __init__(self):
        super().__init__()

    def customized_UI(self):
        self.setWindowTitle('Maximal Patch Analyser')
        self.filename = '../datas/mplog.txt'

        # create two plot
        self.max_plot = pg.PlotWidget()
        self.min_plot = pg.PlotWidget()
        self.hlayout.addWidget(self.max_plot)
        self.hlayout.addWidget(self.min_plot)

    def plot_data(self):
        self.max_plot.clear()
        self.min_plot.clear()

        self.colors = ColorTool(len(self.current_data))
        self.legend = self.max_plot.addLegend()
        for i, line in enumerate(self.current_data):
            method_name = line[2]
            maxacc = line[3:].astype(float)
            minacc = maxacc[len(maxacc) // 2:]
            maxacc = maxacc[:len(maxacc) // 2]
            x = np.array([1, 2, 3, 5, 10, 20])
            pen = self.colors.get_color(i)
            pi1 = self.max_plot.plot(x, maxacc, pen=pen, name=method_name)
            pi2 = self.min_plot.plot(x, minacc, pen=pen, name=method_name)
            # self.legend.addItem(pi1, method_name)


if __name__ == '__main__':
    analyse_main(MaximalPatchAnalyser)



