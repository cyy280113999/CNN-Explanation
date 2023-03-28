from BaseAnalyser import *


class YourAnalyser(BaseAnalyser):
    def __init__(self):
        super().__init__()

    def customized_UI(self):
        # ==== global setting
        self.setWindowTitle('Your Analyser')
        self.filename = '../datas/yourlog.txt'

        # ==== for a plot evaluator, create a plot canvas
        self.plot_widget = pg.PlotWidget()
        # self.plot_widget.showGrid(x=True, y=True)
        # self.plot_widget.setLabel('left', 'Y')
        # self.plot_widget.setLabel('bottom', 'X')
        self.hlayout.addWidget(self.plot_widget)

    def plot_data(self):
        # === drawing
        self.plot_widget.clear()
        self.colors = ColorTool(len(self.current_data))
        self.legend = self.plot_widget.addLegend()
        for i, line in enumerate(self.current_data):
            method_name = line[2]
            # acc = line[3].astype(float)
            pen = self.colors.colors[i]
            # self.plot_widget.plot(x, y, pen=pen, name=method_name)


if __name__ == '__main__':
    analyse_main(YourAnalyser)