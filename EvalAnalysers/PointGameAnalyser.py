from BaseAnalyser import *
import torch


class PointGameAnalyser(BaseAnalyser):
    def __init__(self):
        super().__init__()

    def customized_UI(self):
        self.setWindowTitle('Point Game Analyser')
        self.filename = '../datas/pglog.txt'
        self.remain_ratios = torch.arange(0.05, 1, 0.05)  # L <= v < R
        # create two plot
        self.acc_plot = pg.PlotWidget()
        self.hlayout.addWidget(self.acc_plot)

    def plot_data(self):
        self.acc_plot.clear()

        self.colors = ColorTool(len(self.current_data))
        self.legend = self.acc_plot.addLegend()
        for i, line in enumerate(self.current_data):
            method_name = line[2]
            acc = line[3:].astype(float)
            pen = self.colors.get_color(i)
            self.acc_plot.plot(self.remain_ratios, acc, pen=pen, name=method_name)
            # self.legend.addItem(pi1, method_name)


if __name__ == '__main__':
    analyse_main(PointGameAnalyser)



