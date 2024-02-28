import random

import numpy as np
import os
import sys
import torch
import torch.utils.data as TD
import torchvision
from PIL import Image
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout, QPushButton, QLineEdit, QApplication, QComboBox, \
    QFileDialog

from datasets.OnlyImages import OnlyImages

device = 'cuda'


def generate_abs_filename(this, fn):
    current_file_path = os.path.abspath(this)
    current_directory = os.path.dirname(current_file_path)
    file_to_read = os.path.join(current_directory, fn)
    return file_to_read


# ========== pil image loading
def pilOpen(filename):
    return Image.open(filename).convert('RGB')


# =========== image process
toRRC = torchvision.transforms.RandomResizedCrop(224, scale=(0.25, 1), ratio=(1, 1))

toTensor = torchvision.transforms.ToTensor()

toTensorS224 = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    toTensor,
])

toPIL = torchvision.transforms.ToPILImage()

ImgntMean = [0.485, 0.456, 0.406]
ImgntStd = [0.229, 0.224, 0.225]
ImgntMeanTensor = torch.tensor(ImgntMean).reshape(1, -1, 1, 1)
ImgntStdTensor = torch.tensor(ImgntStd).reshape(1, -1, 1, 1)

# that is compatible for both cpu and cuda
toStd = torchvision.transforms.Normalize(ImgntMean, ImgntStd)


def invStd(tensor):
    tensor = tensor * ImgntStdTensor + ImgntMeanTensor
    return tensor


# ============ std image loading
# image for raw (PIL,numpy) image. x for standardized tensor
def get_image_x(filename='cat_dog_243_282.png', image_folder='input_images/', device=device):
    # require folder , pure filename
    filename = image_folder + filename
    img_PIL = pilOpen(filename)
    img_tensor = toTensorS224(img_PIL).unsqueeze(0)
    img_tensor = toStd(img_tensor).to(device)
    return img_tensor


# =========== ImageNet loading
if sys.platform == 'win32':
    imageNetDefaultDir = r'F:/DataSet/imagenet/'
elif sys.platform == 'linux':
    imageNetDefaultDir = r'/home/dell/datasets/imgnt/'
else:
    imageNetDefaultDir = None
imageNetSplits = {
    'train': 'train/',
    'val': 'val/',
}

default_transform = torchvision.transforms.Compose([
    toTensorS224,
    toStd
])


def loadImageNetClasses(path=imageNetDefaultDir):
    import json
    filename = path + 'imagenet_class_index.json'
    with open(filename) as f:
        c = json.load(f)
        c = {int(i): v[-1] for i, v in c.items()}
        return c


def getImageNet(split, transform=default_transform):
    return torchvision.datasets.ImageNet(root=imageNetDefaultDir,
                                         split=split,
                                         transform=transform)


class DatasetTraveller:
    def __init__(self, dataset):
        super().__init__()
        self.dataSet = dataset
        self.dataSetLen = len(dataset)
        self.img = None
        self.index = 0
        import time
        np.random.seed(int(time.time()))
        self.check = lambda x: x % len(self.dataSet)

    def get(self, i=None):
        if i is not None:
            self.index = self.check(i)
        return self.dataSet[self.index]

    def next(self):
        self.index = self.check(self.index + 1)
        return self.dataSet[self.index]

    def back(self):
        self.index = self.check(self.index - 1)
        return self.dataSet[self.index]

    def rand(self):
        i = np.random.randint(0, self.dataSetLen - 1, (1,))[0]
        self.index = self.check(i)
        return self.dataSet[self.index]


class SubSetFromIndices(TD.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)


class SingleImageLoader(QWidget):
    img=None
    def __init__(self, sendCallBack=None):
        super().__init__()
        self.init_ui()
        self.open_btn.clicked.connect(self.open)
        self.link(sendCallBack)

    def init_ui(self):
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.img_info = QLabel()
        self.set_info()
        self.main_layout.addWidget(self.img_info)
        self.open_btn = QPushButton('Open')
        self.main_layout.addWidget(self.open_btn)

    def set_info(self, p='None'):
        self.img_info.setText(f'Image Info: {p}')

    def open(self):
        filename_long, f_type = QFileDialog.getOpenFileName(directory="./")
        if filename_long:
            self.img = pilOpen(filename_long)
            self.img = toTensor(self.img)
            self.set_info(os.path.basename(filename_long))
            if self.send is not None:
                self.send(self.img)
        else:
            self.img=None
            self.set_info()
            pass

    def link(self, sendCallBack):
        self.send = None
        if sendCallBack is not None:
            self.send = sendCallBack

class FoldImageLoader(QWidget):
    dataSet = None
    index = 0
    img = None
    def __init__(self, sendCallBack=None):
        super().__init__()
        self.initUI()
        self.open_btn.clicked.connect(self.open)
        self.next.clicked.connect(self.indexNext)
        self.back.clicked.connect(self.indexBack)
        self.randbtn.clicked.connect(self.indexRand)
        self.index_editer.returnPressed.connect(self.parse_index)
        self.link(sendCallBack)

    def initUI(self):
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.datasetInfo = QLabel()
        self.set_dataset_info()
        self.main_layout.addWidget(self.datasetInfo)
        self.imgInfo = QLabel()
        self.set_image_info()
        self.main_layout.addWidget(self.imgInfo)
        hlayout = QHBoxLayout()  # add row
        self.open_btn = QPushButton("Open")
        self.back = QPushButton("Back")
        self.next = QPushButton("Next")
        self.randbtn = QPushButton("Rand")
        self.index_editer = QLineEdit("0")
        hlayout.addWidget(self.open_btn)
        hlayout.addWidget(self.back)
        hlayout.addWidget(self.next)
        hlayout.addWidget(self.randbtn)
        hlayout.addWidget(self.index_editer)
        self.main_layout.addLayout(hlayout)
        self.open_btn.setMinimumHeight(40)
        self.back.setMinimumHeight(40)
        self.next.setMinimumHeight(40)
        self.randbtn.setMinimumHeight(40)
        self.index_editer.setMinimumHeight(40)
        self.index_editer.setMaximumWidth(80)
        self.index_editer.setMaxLength(8)
        self.btns = [self.back, self.next, self.randbtn, self.index_editer]
        self.show_btns(False)

    def set_dataset_info(self):
        if self.dataSet is None:
            self.datasetInfo.setText(f"Please open.")
        elif len(self.dataSet) == 0:
            self.datasetInfo.setText(f"No image in folder.")
        else:
            self.datasetInfo.setText(f"Images Index From 0 to {len(self.dataSet) - 1}.")

    def set_image_info(self, p='None'):
        self.imgInfo.setText(f'Image Info: {p}')

    def show_btns(self, flag=True):
        if flag:
            func = lambda x: x.show()
        else:
            func = lambda x: x.hide()
        for b in self.btns:
            func(b)

    def getImage(self):
        self.img, c = self.dataSet[self.index]
        p, _ = self.dataSet.samples[self.index]
        self.set_image_info(os.path.basename(p))
        self.img = toTensor(self.img)
        if self.send is not None:
            self.send(self.img)

    def set_index(self, i):
        i = i % len(self.dataSet)
        self.index = i
        self.index_editer.setText(str(i))
        self.getImage()

    def parse_index(self):
        t = self.index_editer.text()
        try:
            i = int(t)
            self.set_index(i)
        except Exception() as e:
            pass

    def open(self):
        directory = QFileDialog.getExistingDirectory(directory="./")
        if directory:
            self.dataSet = OnlyImages(directory)
            self.set_dataset_info()
            if len(self.dataSet) == 0:
                self.show_btns(False)
            else:
                self.show_btns()
                self.set_index(0)

    def indexNext(self):
        self.set_index(self.index + 1)

    def indexBack(self):
        self.set_index(self.index - 1)

    def indexRand(self):
        self.set_index(random.randint(0, len(self.dataSet) - 1) + 1)

    def link(self, sendCallBack):
        self.send = None
        if sendCallBack is not None:
            self.send = sendCallBack


class TorchDatesetLoader(QWidget):
    dataSet = None
    index = 0
    img = None
    def __init__(self,sendCallBack=None):
        super().__init__()
        self.initUI()
        self.next.clicked.connect(self.indexNext)
        self.back.clicked.connect(self.indexBack)
        self.randbtn.clicked.connect(self.indexRand)
        self.index_editer.returnPressed.connect(self.parse_index)
        self.link(sendCallBack)

    def initUI(self):
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.datasetInfo = QLabel()
        self.set_dataset_info()
        self.main_layout.addWidget(self.datasetInfo)
        self.imgInfo = QLabel()
        self.set_image_info()
        self.main_layout.addWidget(self.imgInfo)
        hlayout = QHBoxLayout()  # add row
        self.back = QPushButton("Back")
        self.next = QPushButton("Next")
        self.randbtn = QPushButton("Rand")
        self.index_editer = QLineEdit("0")
        hlayout.addWidget(self.back)
        hlayout.addWidget(self.next)
        hlayout.addWidget(self.randbtn)
        hlayout.addWidget(self.index_editer)
        self.main_layout.addLayout(hlayout)
        self.back.setMinimumHeight(40)
        self.next.setMinimumHeight(40)
        self.randbtn.setMinimumHeight(40)
        self.index_editer.setMinimumHeight(40)
        self.index_editer.setMaximumWidth(80)
        self.index_editer.setMaxLength(8)


    def set_dataset_info(self):
        if self.dataSet is None:
            self.datasetInfo.setText(f"Please open.")
        elif len(self.dataSet) == 0:
            self.datasetInfo.setText(f"No image in folder.")
        else:
            self.datasetInfo.setText(f"Images Index From 0 to {len(self.dataSet) - 1}.")

    def set_image_info(self, p='None'):
        self.imgInfo.setText(f'Image Info: {p}')

    def getImage(self):
        self.img, c = self.dataSet[self.index]
        p, _ = self.dataSet.samples[self.index]
        self.set_image_info(os.path.basename(p))
        self.img = toTensor(self.img)
        if self.send is not None:
            self.send(self.img)

    def set_index(self, i):
        i = i % len(self.dataSet)
        self.index = i
        self.index_editer.setText(str(i))
        self.getImage()

    def parse_index(self):
        t = self.index_editer.text()
        try:
            i = int(t)
            self.set_index(i)
        except Exception() as e:
            pass

    def indexNext(self):
        self.set_index(self.index + 1)

    def indexBack(self):
        self.set_index(self.index - 1)

    def indexRand(self):
        self.set_index(random.randint(0, len(self.dataSet) - 1) + 1)

    def set_dateset(self,ds):
        self.dataSet=ds
        self.set_dataset_info()
        self.set_index(0)

    def link(self, sendCallBack):
        self.send = None
        if sendCallBack is not None:
            self.send = sendCallBack