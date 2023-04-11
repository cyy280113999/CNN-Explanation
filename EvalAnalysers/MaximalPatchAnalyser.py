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

        self.x = np.array([1, 2, 3, 5, 10, 20])

    def plot_data(self):
        self.max_plot.clear()
        self.min_plot.clear()

        self.pens = PenTool(len(self.current_data))
        self.legend = self.max_plot.addLegend()
        for i, line in enumerate(self.current_data):
            method_name = line[2]
            maxacc = line[3:].astype(float)
            minacc = maxacc[len(maxacc) // 2:]
            maxacc = maxacc[:len(maxacc) // 2]
            pen = self.pens.pens[i]
            self.max_plot.plot(self.x, maxacc, pen=pen, name=method_name)
            self.min_plot.plot(self.x, minacc, pen=pen, name=method_name)


if __name__ == '__main__':
    analyse_main(MaximalPatchAnalyser)



