import os
from math import floor,ceil

import torch
import torchvision
import xml.etree.ElementTree as ET
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

default_transform=torchvision.transforms.Compose(transforms=[
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

default_root='F:/DataSet/imagenet/'
default_bbox_root='F:/DataSet/imagenet/ILSVRC2012_bbox_val_v3/val/'

def clip(x, lower, upper):
    return max(lower, min(x, upper))

def BBoxCenterCrop(bbox,h,w):
    # w->x
    xmin, ymin, xmax, ymax = bbox
    if h>w:
        resize_ratio = 224/w
        xmin = floor(xmin*resize_ratio)
        xmax = ceil(xmax*resize_ratio)
        cut_edge = (h*resize_ratio-224)/2
        ymin = clip(floor(ymin*resize_ratio-cut_edge),0,224)
        ymax = clip(ceil(ymax*resize_ratio-cut_edge),0,224)
    else:
        resize_ratio = 224 / h
        ymin = floor(ymin * resize_ratio)
        ymax = ceil(ymax * resize_ratio)
        cut_edge = (w * resize_ratio - 224) / 2
        xmin = clip(floor(xmin * resize_ratio - cut_edge),0, 224)
        xmax = clip(ceil(xmax * resize_ratio - cut_edge),0, 224)
    return [xmin,ymin,xmax,ymax]

# bbox imagenet
class BBImgnt(torchvision.datasets.ImageNet):
    def __init__(self,
                 root=default_root,
                 bbox_root=default_bbox_root,
                 transform=default_transform):
        super(BBImgnt, self).__init__(root=root,
                                      split='val',
                                      transform=default_transform)
        self.bbox_root = bbox_root

    def __getitem__(self, index):
        path, target = self.samples[index]
        path=path.replace('\\', '/')
        pure_filename = path.split('/')[-1].split('.')[0]
        # get x
        sample = self.loader(path)
        h,w = sample.size
        sample = self.transform(sample)
        # get bboxs
        ananame = os.path.join(self.bbox_root,pure_filename + '.xml')
        tree = ET.parse(ananame)
        root = tree.getroot()
        bboxs = []
        for neighbor in root.iter('bndbox'):
            xmin = int(neighbor.find('xmin').text)
            ymin = int(neighbor.find('ymin').text)
            xmax = int(neighbor.find('xmax').text)
            ymax = int(neighbor.find('ymax').text)
            bbox = [xmin, ymin, xmax, ymax]
            bbox = BBoxCenterCrop(bbox,h,w)
            bboxs.append(bbox)

        return sample, target, bboxs

if __name__ == '__main__':
    import numpy as np
    import torch.utils.data as TD
    from tqdm import tqdm

    np.random.seed(1)
    torch.random.manual_seed(1)

    num_samples = 1000
    ds=BBImgnt()
    indices = np.random.choice(len(ds), num_samples)
    subSet = TD.Subset(ds, indices)
    dataLoader = TD.DataLoader(subSet, batch_size=1, pin_memory=True, num_workers=0)
    num_samples = len(dataLoader)

    with torch.no_grad():
        for data in tqdm(dataLoader):
            x,y,bboxs = data
            for b in bboxs:
                print(b)
            break


