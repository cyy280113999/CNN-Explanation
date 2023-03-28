from BaseAnalyser import *
from pyqtgraph import PlotDataItem, BarGraphItem


class ProbChangeAnalyser(BaseAnalyser):
    def __init__(self):
        super().__init__()

    def customized_UI(self):
        self.setWindowTitle('Prob Change Analyser')
        self.filename = '../datas/pclog.txt'

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Accuracy')
        self.plot_widget.setLabel('bottom', 'Method')
        self.hlayout.addWidget(self.plot_widget)

    def plot_data(self):
        self.plot_widget.clear()
        self.colors = ColorTool(len(self.current_data))
        self.legend = self.plot_widget.addLegend()
        for i, line in enumerate(self.current_data):
            method_name = line[2]
            acc = line[3].astype(float)
            std = line[4].astype(float)
            pen = self.colors.colors[i]
            dot_item = PlotDataItem(x=[i],y=[acc],pen='k', symbol='o')
            bar_item = BarGraphItem(x=[i],y=[acc], height=[std], width=0.8, name=method_name, brush=pen)
            self.plot_widget.addItem(bar_item)
            self.plot_widget.addItem(dot_item)


if __name__ == '__main__':
    analyse_main(ProbChangeAnalyser)