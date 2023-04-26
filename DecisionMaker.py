# user
import torch
from torchvision.models import VGG

from methods.LIDDecomposer import LIDDecomposer, relevanceFindByName
from utils import *

# torch initial
device = "cuda"

from ExpVis import ExplainMethodSelector, DataSetLoader, MainWindow, imageNetVal


class DecisionMaker(ExplainMethodSelector):
    def backward_feature(self, g):
        for m in self.model.features[::-1]:
            g = self.lid.backward_baseunit(m, g)
        return g

    def generateHeatmaps(self):
        if self.img is None or self.model is None:
            return
        assert isinstance(self.model, VGG)
        try:
            classes = list(int(cls) for cls in self.classSelector.text().split(','))
        except Exception as e:
            self.predictionScreen.setPlainText(e.__str__())
            return

        cls = classes[0]
        self.lid = LIDDecomposer(self.model, LINEAR=False)
        self.lid.forward(self.img_dv)
        self.lid.backward(cls, 'sig')
        g = self.model.features[-1].g
        hm = relevanceFindByName(self.model.features, -1)

        channel_scores = hm.sum(dim=[2, 3]).flatten()
        ranking = channel_scores.argsort(descending=True)
        feature_count = 9
        # grouped features
        # feature_size = int(len(channel_scores) / feature_count)
        # feature_indices = [ranking[i * feature_size:i * feature_size + feature_size] for i in range(feature_count)]
        # top 6 & bottom 3 features
        feature_size = 1
        feature_indices = [[i] for i in ranking[:6].tolist()+ranking[-3:].tolist()]

        activation_maps = []
        for i in range(9):
            activation_maps.append(self.model.features[-1].y[1, feature_indices[i]].sum(0, True))
        activation_maps = torch.vstack(activation_maps)
        activation_maps = activation_maps / activation_maps.abs().max()

        relevance_maps = []
        for i in range(9):
            feature_g = torch.zeros_like(g)
            feature_g[0, feature_indices[i]] = g[0, feature_indices[i]]
            self.backward_feature(feature_g)
            pixelmap = relevanceFindByName(self.model.features, 4).sum(1, True)
            relevance_maps.append(pixelmap)
        relevance_maps = torch.vstack(relevance_maps)
        relevance_maps = relevance_maps / relevance_maps.abs().max()

        self.imageCanvas.pglw.clear()
        row_count = 3
        for i, cls in enumerate(range(9)):
            # ___________runningCost___________.tic()
            row = i / row_count
            col = i % row_count
            pi = self.imageCanvas.pglw.addPlot(row=row, col=col)  # 2 images
            plotItemDefaultConfig(pi)

            # show activation
            whattoshow=activation_maps[i]

            # show pixel relevance map
            #whattoshow = relevance_maps[i]

            pi.addItem(pg.ImageItem(toPlot(whattoshow), levels=[-1, 1], lut=lrp_lut))
            pi.setTitle(f'{channel_scores[feature_indices[i]].sum()}')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    imageLoader = DataSetLoader()
    explainMethodsSelector = DecisionMaker()
    imageCanvas = ImageCanvas()
    mw = MainWindow(imageLoader, explainMethodsSelector, imageCanvas, SeperatedCanvas=False)
    # initial settings
    explainMethodsSelector.init(mw.imageChangeSignal, imageNetVal, canvas=imageCanvas)
    imageLoader.init(mw.imageChangeSignal)
    mw.show()
    sys.exit(app.exec_())
