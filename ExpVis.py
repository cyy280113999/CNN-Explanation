# sys
# nn
# gui
from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QGroupBox, QMainWindow

from HeatmapMethods import heatmap_methods
# user
from utils import *

# draw

# torch initial
device = "cuda"
# torch.backends.cudnn.benchmark=True
# torch.set_grad_enabled(False)

from datasets.DiscrimDataset import DiscrimDataset

imageNetVal = getImageNet('val', show_transform)


class DataSetLoader(QGroupBox):
    def __init__(self):
        super().__init__()
        # key: dataset name , value: is folder or not
        self.classes = loadImageNetClasses()

        class EnumDS:
            CusImage = "Customized Image"
            CusFolder = "Customized Folder"
            ImgVal = "ImageNet Val"
            ImgTrain = "ImageNet Train"
            Discrim = "Discrim DataSet"

        self.available_datasets = EnumDS()
        self.dataSets = {
            self.available_datasets.CusImage: None,
            self.available_datasets.CusFolder: None,
            self.available_datasets.ImgVal: lambda: imageNetVal,
            self.available_datasets.ImgTrain: lambda: getImageNet('train', None),
            self.available_datasets.Discrim: lambda: DiscrimDataset(transform=None, MultiLabel=False),
        }
        self.dataSet = None
        self.index = 0
        self.img_raw = None
        self.img = None
        self.modes = {
            "None": None,
            "Corner Mask": lambda im: im * cornerMask(im, r=40),
            "AddNoise 001Std": lambda im: im + 0.01 * torch.randn_like(im),
            "AddNoise 003Std": lambda im: im + 0.03 * torch.randn_like(im),
            "AddNoise 01Std": lambda im: im + 0.1 * torch.randn_like(im),
        }
        self.imageMode = None
        self.init_ui()
        # actions
        # def refresh(self,x=None):
        #
        # self.dataSetLen.refresh=refresh
        self.dataSetSelect.currentIndexChanged.connect(self.dataSetChange)
        self.single_loader.link(self.set_image)
        self.folder_loader.link(self.set_image)
        self.td_loader.link(self.set_image)

        # self.rrcbtn.clicked.connect(lambda :self.rrcbtn.setChecked(not self.rrcbtn.isChecked()))
        self.modeSelects.currentIndexChanged.connect(self.modeChange)
        self.regeneratebtn.clicked.connect(self.imageChange)

        self.checkIndex = lambda x: x % len(self.dataSet)

    def init_ui(self):
        # self.setMaximumWidth(600)
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.setTitle("Image Loader")

        # data set
        hlayout = QHBoxLayout()
        self.dataSetSelect = DictComboBox(self.dataSets)
        # self.dataSetSelect.resize(200,40)
        hlayout.addWidget(QLabel("Data Set: "))
        hlayout.addWidget(self.dataSetSelect)
        self.main_layout.addLayout(hlayout)

        # image choose
        hlayout = QHBoxLayout()
        self.single_loader=SingleImageLoader()
        self.folder_loader=FoldImageLoader()
        self.td_loader=TorchDatesetLoader()

        hlayout.addWidget(self.single_loader)
        hlayout.addWidget(self.folder_loader)
        hlayout.addWidget(self.td_loader)
        self.main_layout.addLayout(hlayout)

        # new line
        hlayout = QHBoxLayout()
        # rrc switch
        self.rrcbtn = QPushButton("RRC")
        self.rrcbtn.setMinimumHeight(40)
        self.rrcbtn.setCheckable(True)
        # image modify after tostd
        # interface: im->im

        self.modeSelects = DictComboBox(self.modes)
        hlayout.addWidget(QLabel('Image modify:'))
        hlayout.addWidget(self.rrcbtn)
        hlayout.addWidget(self.modeSelects)

        # re-
        self.regeneratebtn = QPushButton("ReGenerate")
        self.regeneratebtn.setMinimumHeight(40)
        hlayout.addWidget(self.regeneratebtn)
        self.main_layout.addLayout(hlayout)

        # image show
        self.imageCanvas = ImageCanvas()
        # self.imageCanvas.showImage(loadTestImg())
        self.main_layout.addWidget(self.imageCanvas)

        self.loaders = [self.single_loader, self.folder_loader, self.td_loader]
        self.showLoader(self.single_loader)

    def showLoader(self, which):
        for l in self.loaders:
            l.hide()
        which.show()

    def dataSetChange(self):
        t = self.dataSetSelect.currentText()
        if t == self.available_datasets.CusFolder:
            self.showLoader(self.folder_loader)
        elif t == self.available_datasets.CusImage:
            self.showLoader(self.single_loader)
        else:
            self.td_loader.set_dateset(self.dataSets[t]())
            self.showLoader(self.td_loader)

    def modeChange(self):
        t = self.modeSelects.currentText()
        self.imageMode = self.modes[t]
        self.imageChange()

    def imageChange(self):
        if self.img_raw is not None:
            self.img = toPIL(self.img_raw.squeeze(0))  # ready to crop/resize
            self.img = toTensorPad224(self.img).unsqueeze(0)  # (1,3,224,224) [0,1]
            if self.rrcbtn.isChecked():
                self.img = toRRC(self.img)
            if self.imageMode:
                self.img = self.imageMode(self.img).clip(min=0, max=1)  # [0,1]
            self.imageCanvas.showImage(toPlot(self.img))
            self.sendImg(self.img)

    def sendImg(self, x):
        if hasattr(self, 'emitter') and self.emitter is not None:
            self.emitter.emit(x)

    def init(self, send_signal: pyqtSignal):
        self.emitter = send_signal

    def set_image(self, x):
        self.img_raw=x
        self.imageChange()


class ExplainMethodSelector(QGroupBox):
    def __init__(self):
        super(ExplainMethodSelector, self).__init__()
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        # self.setMaximumWidth(600)
        self.setTitle("Explain Method Selector")

        # -- settings
        self.pred_topk = 20
        self.max_count_heatmap = 6
        showed_models = [
            model_names.vgg16,
            model_names.alexnet,
            model_names.res18,
            model_names.res34,
            model_names.res50,
            model_names.res101,
            model_names.res152,
            model_names.googlenet,
        ]
        self.models = {
            # "None": lambda: None,
            name: get_model_caller(name) for name in showed_models
        }
        self.model = None

        self.methods = heatmap_methods

        # the mask interface, all masks must follow this:
        # the masked heatmap generated by calling "im, cover = m(im, hm)"
        # the param hm is raw heatmap, the im is input image
        # the output im is a valid printable masked heatmap image
        # the output hm is raw heatmap
        # mask: im, hm -> image[0,1], covered heatmap[-1,1]
        self.masks = {
            "Raw Heatmap": lambda im, hm: (None, hm),
            "Overlap": lambda im, hm: (im, hm),
            "Reversed Image": lambda im, hm: ((hm + 1) / 2, None),
            "Positive Overlap": lambda im, hm: (im * positize(hm), None),
            "1Std Overlap": lambda im, hm: (im * (hm > (hm.mean() + hm.std())), None),
            "Positive 1Std": lambda im, hm: (im * positize(hm), None),
            "Sparsity 50": lambda im, hm: (im * binarize(hm, sparsity=0.5), None),
            "Maximal Patch": lambda im, hm: (im * maximalPatch(hm, top=True, r=20), None),
            "Minimal Patch": lambda im, hm: (im * maximalPatch(hm, top=False, r=20), None),
            "Corner Mask": lambda im, hm: (im * cornerMask(hm, r=40), None),
            # "AddNoise 0.1Std": lambda im, hm: (im + 0.1 * torch.randn_like(im), None),
            # "AddNoise 0.5Std": lambda im, hm: (im + 0.5 * torch.randn_like(im), None),
            "AddNoiseN0.5Std 50% ": lambda im, hm: (im + binarize_add_noisy_n(hm, top=True, std=0.5), None),
            "AddNoiseN1Std 50% ": lambda im, hm: (im + binarize_add_noisy_n(hm, top=True, std=1), None),
            "AddNoiseN0.5Std Inv50% ": lambda im, hm: (im + binarize_add_noisy_n(hm, top=False, std=0.5), None),
            "AddNoiseN1Std Inv50% ": lambda im, hm: (im + binarize_add_noisy_n(hm, top=False, std=1), None),
        }

        self.method = None
        self.img = None

        self.hms = []
        self.mask = None

        hlayout = QHBoxLayout()
        self.modelSelect = ListComboBox(showed_models)
        hlayout.addWidget(TippedWidget("Model: ", self.modelSelect))
        self.main_layout.addLayout(hlayout)

        self.methodSelect = ListWidget(self.methods.keys())
        # hlayout.addWidget(TippedWidget("Method: ", self.methodSelect))

        hlayout = QHBoxLayout()
        self.maskSelect = ListComboBox(self.masks.keys())
        hlayout.addWidget(TippedWidget("Mask: ", self.maskSelect))

        self.regeneratebtn1 = QPushButton("Reload Image")
        self.regeneratebtn1.setMinimumHeight(40)
        hlayout.addWidget(self.regeneratebtn1)
        # re-
        self.regeneratebtn2 = QPushButton("ReGenerate Heatmap")
        self.regeneratebtn2.setMinimumHeight(40)
        hlayout.addWidget(self.regeneratebtn2)
        self.main_layout.addLayout(hlayout)

        # class
        self.classSelector = QLineEdit("")
        self.classSelector.setMinimumHeight(40)
        # self.classSelector.setMaximumWidth(500)
        self.classSelector.setPlaceholderText("classes choose:")
        self.main_layout.addWidget(TippedWidget("Classes:", self.classSelector))

        # class sort
        hlayout = QHBoxLayout()
        self.autoClass = QPushButton("Auto Class")
        self.autoClass.setMinimumHeight(40)
        self.autoClass.setCheckable(True)
        self.autoClass.setChecked(True)
        hlayout.addWidget(QLabel(f"Prediction Top {self.pred_topk}"))
        hlayout.addWidget(self.autoClass)
        self.addexpbtn = QPushButton("Examples")
        self.addexpbtn.setMinimumHeight(40)
        self.addexpbtn.setCheckable(True)
        hlayout.addWidget(self.addexpbtn)
        self.main_layout.addLayout(hlayout)

        hlayout = QHBoxLayout()
        self.predictionScreen = QPlainTextEdit("classes prediction shown,example:\n1, 0.5, Cat")
        self.predictionScreen.setMinimumHeight(40)
        # self.predictionScreen.setMaximumWidth(800)
        self.predictionScreen.setReadOnly(True)

        vlayout=QVBoxLayout()
        vlayout.addWidget(QLabel("Method: "))
        vlayout.addWidget(self.methodSelect)

        hlayout.addWidget(self.predictionScreen, 2)
        hlayout.addLayout(vlayout, 1)
        self.main_layout.addLayout(hlayout)

        # actions
        self.modelSelect.currentIndexChanged.connect(self.modelChange)
        # self.methodSelect.currentIndexChanged.connect(self.methodChange)
        self.methodSelect.itemSelectionChanged.connect(self.methodChange)
        self.maskSelect.currentIndexChanged.connect(self.maskChange)

        self.regeneratebtn1.clicked.connect(lambda: self.generatePrediction())
        self.classSelector.returnPressed.connect(self.generateHeatmaps)
        self.regeneratebtn2.clicked.connect(self.generateHeatmaps)

    def init(self, receive_signal, imgnt, canvas=None):
        # output canvas
        if canvas is not None:
            self.imageCanvas = canvas
        else:
            self.imageCanvas = ImageCanvas()  # no add
        self.imgnt = imgnt
        self.imgntClassNames = loadImageNetClasses()
        self.reciever = receive_signal
        self.reciever.connect(self.generatePrediction)
        self.modelChange()
        self.maskChange()

    # def outputCanvasWidget(self):
    #     return self.imageCanvas

    def modelChange(self):
        t = self.modelSelect.currentText()
        self.model = self.models[t]
        if self.model is not None:
            self.model = self.model()
            self.methodChange()

    def methodChange(self):
        t = self.methodSelect.currentItem().text()
        self.method = self.methods[t]
        if self.method is not None and self.model is not None:
            self.method = self.method(self.model)
            self.generatePrediction()

    def generatePrediction(self, x=None):
        if x is not None:
            self.img = x
        if self.img is None:
            return
        # test, send img to canvas
        # self.imageCanvas.showImage(ToPlot(InverseStd(self.img)))
        if self.model is not None:
            self.tsr = toStd(self.img).to(device).requires_grad_()
            if self.autoClass.isChecked():
                # predict
                prob = self.model(self.tsr).softmax(1)
                topk = prob.sort(1, descending=True)
                topki = topk[1][0, :self.pred_topk]
                topkv = topk[0][0, :self.pred_topk]
                # show info
                if self.imgntClassNames is None:
                    raise Exception("saveClasses First.")
                self.predictionScreen.setPlainText(
                    "\n".join(self.PredInfo(i.item(), v.item()) for i, v in zip(topki, topkv)))
                self.classSelector.setText(",".join(str(i.item()) for i in topki[:self.max_count_heatmap]))
            else:
                self.predictionScreen.setPlainText('No Prediction.')
            self.generateHeatmaps()

    def PredInfo(self, cls, prob=None, max_len=30):
        if prob:
            s = f"{cls}:\t{prob:.4f}:\t{self.imgntClassNames[cls]}"
        else:
            s = f"{cls}:{self.imgntClassNames[cls]}"
            if len(s) > max_len:
                s = s[:max_len]
            else:
                s = s + (max_len - len(s)) * '_'
        return s

    def generateHeatmaps(self):
        if self.img is None or self.model is None or self.method is None:
            return
        try:
            # get classes
            self.classes = list(int(cls) for cls in self.classSelector.text().split(','))
            self.classes = self.classes[:self.max_count_heatmap]  # always 6
            # classes = classes[:6]
        except Exception as e:
            self.predictionScreen.setPlainText(f'{e.__str__()}\nplease give class in chooser split by ","')
            return
        # heatmaps
        self.hms = []
        with torch.enable_grad():
            for cls in self.classes:
                self.hms.append(self.method(self.tsr, cls).detach().cpu())
        self.generatePlots()

    def maskChange(self):
        t = self.maskSelect.currentText()
        self.mask = self.masks[t]
        if self.mask is not None:
            self.generatePlots()

    def dynamicSparsity(self):
        self.spartisy_timer = QTimer()
        []

    def sparsityEvent(self):
        []

    def generatePlots(self):
        if self.hms is None or not self.hms:
            return
        add_exp = self.addexpbtn.isChecked()
        self.imageCanvas.pglw.clear()
        row_count = 2
        for i, cls in enumerate(self.classes):
            row = i // row_count
            col = i % row_count
            # self.imageCanvas.pglw.nextRow()
            l = self.imageCanvas.pglw.addLayout(row=row, col=col)  # 2 images
            hm = self.hms[i]

            pi: pg.PlotItem = l.addPlot(0, 0)
            plotItemDefaultConfig(pi)
            p = None
            if self.mask is not None:
                masked, covering = self.mask(self.img, hm)
                opac = 1.
                if masked is not None:
                    p = self.maskScore(masked, cls)
                    pi.addItem(pg.ImageItem(toPlot(masked), levels=[0, 1], opacity=opac))
                    opac = 0.7  # if image exist, decrease the opacity of covering
                if covering is not None:
                    pi.addItem(pg.ImageItem(toPlot(covering),
                                            # compositionMode=QPainter.CompositionMode.CompositionMode_Overlay,
                                            levels=[-1, 1], lut=lrp_lut, opacity=opac))
            else:  # mask not initialized
                pi.addItem(pg.ImageItem(toPlot(hm), levels=[-1, 1], lut=lrp_lut))
            pi.setTitle(self.PredInfo(cls, p, 50))
            if add_exp:
                example = self.cls_example(cls)
                im_exp = toPlot(toTensorPad224(example))
                pexp = l.addPlot(0, 1)
                plotItemDefaultConfig(pexp)
                pexp.addItem(pg.ImageItem(im_exp))

    def cls_example(self, cls):
        wnid = self.imgnt.wnids[cls]
        directory = self.imgnt.split_folder
        target_dir = os.path.join(directory, wnid)
        if not os.path.isdir(target_dir):
            raise Exception()
        allImages = list(e.name for e in os.scandir(target_dir))
        imgCount = len(allImages)
        j = random.randint(0, imgCount - 1)
        imgpath = os.path.join(target_dir, allImages[j])
        example = pilOpen(imgpath)
        return example

    def maskScore(self, x, y):
        if self.maskSelect.currentText() in ["Positive Only", "Sparsity 50", "Maximal Patch", "Corner Mask",
                                             "AddNoiseN 0.5", "AddNoiseN 1", "AddNoiseN 0.5 Inv", "AddNoiseN 1 Inv"]:
            return self.getProb(x, y)
        else:
            return None

    def getProb(self, x, y):
        return self.model(toStd(x).to(device)).softmax(1)[0, y]

    def callImgChg(self):
        # 我真的服，signal擅自修改了我的默认参数。这下你添加不了了吧
        self.generatePrediction()

    def sparsityTimer(self):
        []


class MainWindow(QMainWindow):
    # must define signal in class
    imageChangeSignal = pyqtSignal(torch.Tensor)

    def __init__(self, imageLoader, explainMethodsSelector, imageCanvas,
                 SeperatedCanvas=True):
        super().__init__()
        self.imageLoader = imageLoader
        self.explainMethodsSelector = explainMethodsSelector
        self.imageCanvas = imageCanvas

        # set mainFrame UI, main window settings
        self.setWindowTitle("Explaining Visualization")
        # self.setGeometry(200, 100, 1000, 800) #specify window size
        # self.frameGeometry().moveCenter(QDesktopWidget.availableGeometry().center())
        # self.setWindowIcon(QIcon('EXP.ico'))
        # self.setIconSize(QSize(20, 20))

        mainPanel = QWidget()
        self.setCentralWidget(mainPanel)
        control_layout = QHBoxLayout()
        # split window into L&R.
        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()
        mainPanel.setLayout(control_layout)
        control_layout.addLayout(left_panel)
        control_layout.addLayout(right_panel)
        if SeperatedCanvas:
            # add controllers
            left_panel.addWidget(self.imageLoader)
            right_panel.addWidget(self.explainMethodsSelector)
            self.imageCanvas.showMaximized()
            self.show()
        else:
            # left_panel add controllers
            left_panel.addWidget(self.imageLoader, 2)
            left_panel.addWidget(self.explainMethodsSelector, 3)
            # cright_panel add display screen
            right_panel.addWidget(self.imageCanvas)
            control_layout.setStretchFactor(left_panel, 1)
            control_layout.setStretchFactor(right_panel, 4)
            # left_panel.setStretchFactor(1)
            # right_panel.setStretchFactor(4)
            # self.imageLoader.setMaximumWidth(800)
            # self.explainMethodsSelector.setMaximumWidth(800)
            self.showMaximized()


def expVisMain(SeperatedWindow=False):
    global imageNetVal
    # --workflow
    # create window
    create_qapp()
    imageLoader = DataSetLoader()  # keep this instance alive!
    explainMethodsSelector = ExplainMethodSelector()
    imageCanvas = ImageCanvas()
    mw = MainWindow(imageLoader, explainMethodsSelector, imageCanvas, SeperatedCanvas=SeperatedWindow)
    # initial settings
    explainMethodsSelector.init(mw.imageChangeSignal, imageNetVal, canvas=imageCanvas)
    imageLoader.init(mw.imageChangeSignal)
    mw.show()
    loop_qapp()


if __name__ == "__main__":
    expVisMain(False)
