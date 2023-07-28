import numpy as np
import os
import sys
import torch
import torch.utils.data as TD
import torchvision
from PIL import Image


device = 'cuda'

def generate_abs_filename(this,fn):
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

    def check(self, i):
        i = i % len(self.dataSet)
        self.index = i

    def get(self, i=None):
        if i is None:
            i = self.index
        self.check(i)
        return self.dataSet[self.index]

    def next(self):
        self.check(self.index + 1)
        return self.dataSet[self.index]

    def back(self):
        self.check(self.index - 1)
        return self.dataSet[self.index]

    def rand(self):
        i = np.random.randint(0, self.dataSetLen - 1, (1,))[0]
        self.check(i)
        return self.dataSet[self.index]

class SubSetFromIndices(TD.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)