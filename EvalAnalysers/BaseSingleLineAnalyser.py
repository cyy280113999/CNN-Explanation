from BaseAnalyser import *


class BaseSingleLineAnalyser(BaseAnalyser):
    def __init__(self):
        super().__init__()

    def customized_UI(self):
        # create two plot
        self.xy_plot = pg.PlotWidget()
        self.hlayout.addWidget(self.xy_plot)
        self.setting()

    def setting(self):
        raise NotImplementedError()
        # your filename
        # self.filename = filename
        # your x for plot
        # self.x = x
        # your y is with same length as x stored at each line[3:]

    def plot_data(self):
        self.xy_plot.clear()
        self.pens = PenTool(len(self.current_data))
        self.legend = self.xy_plot.addLegend()
        for i, line in enumerate(self.current_data):
            method_name = line[2]
            y = line[3:].astype(float)
            pen = self.pens.pens[i]
            self.xy_plot.plot(self.x, y, pen=pen, name=method_name)


if __name__ == '__main__':
    analyse_main(BaseSingleLineAnalyser)



