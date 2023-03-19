import os
from math import floor, ceil

import torch
import torchvision
import xml.etree.ElementTree as ET


def clip(x, lower, upper):
    return max(lower, min(x, upper))


def BBoxResizeCenterCrop(bbox, origin_size, target_size=None):
    # w->x
    h, w = origin_size
    xmin, ymin, xmax, ymax = bbox
    if h > w:
        resize_ratio = 224 / w
        xmin = clip(round(xmin * resize_ratio), 0, 224)
        xmax = clip(round(xmax * resize_ratio), 0, 224)
        cut_edge = (h * resize_ratio - 224) / 2
        ymin = clip(round(ymin * resize_ratio - cut_edge), 0, 224)
        ymax = clip(round(ymax * resize_ratio - cut_edge), 0, 224)
    else:
        resize_ratio = 224 / h
        ymin = clip(round(ymin * resize_ratio), 0, 224)
        ymax = clip(round(ymax * resize_ratio), 0, 224)
        cut_edge = (w * resize_ratio - 224) / 2
        xmin = clip(round(xmin * resize_ratio - cut_edge), 0, 224)
        xmax = clip(round(xmax * resize_ratio - cut_edge), 0, 224)
    if xmin==xmax:
        if xmin==0:
            xmax=1
        else:
            xmin-=1
    if ymin==ymax:
        if ymin==0:
            ymax=1
        else:
            ymin-=1
    return [xmin, ymin, xmax, ymax]


class BBImgnt(torchvision.datasets.ImageNet):
    """
    bbox imagenet
    bbox = [xmin, ymin, xmax, ymax]
    bboxs=[bbox,]
    sample=(x, y, bboxs)
    """
    default_transform = torchvision.transforms.Compose(transforms=[
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    default_root = 'F:/DataSet/imagenet/'
    default_bbox_root = 'F:/DataSet/imagenet/ILSVRC2012_bbox_val_v3/val/'
    def __init__(self,
                 root=default_root,
                 bbox_root=default_bbox_root,
                 transform=default_transform):
        super(BBImgnt, self).__init__(root=root,
                                      split='val',
                                      transform=transform)
        self.bbox_root = bbox_root

    def __getitem__(self, index):
        path, target = self.samples[index]
        path = path.replace('\\', '/')
        pure_filename = path.split('/')[-1].split('.')[0]
        # get x
        sample = self.loader(path)
        w, h = sample.size  # notice! pil.size:(w, h)
        # get bboxs
        ananame = os.path.join(self.bbox_root, pure_filename + '.xml')
        tree = ET.parse(ananame)
        root = tree.getroot()
        bboxs = []
        for neighbor in root.iter('bndbox'):
            xmin = int(neighbor.find('xmin').text)
            ymin = int(neighbor.find('ymin').text)
            xmax = int(neighbor.find('xmax').text)
            ymax = int(neighbor.find('ymax').text)
            bbox = [xmin, ymin, xmax, ymax]
            bboxs.append(bbox)
        if self.transform:
            sample = self.transform(sample)
            for i, bbox in enumerate(bboxs):
                bboxs[i] = BBoxResizeCenterCrop(bbox, (h, w))
        return sample, target, bboxs


if __name__ == '__main__':
    import numpy as np
    import torch.utils.data as TD
    from tqdm import tqdm

    np.random.seed(1)
    torch.random.manual_seed(1)

    num_samples = 1000
    ds = BBImgnt()
    # indices = np.random.choice(len(ds), num_samples)
    # ds = TD.Subset(ds, indices)
    # dataLoader = TD.DataLoader(ds,shuffle=False, batch_size=1, pin_memory=True, num_workers=0)
    # num_samples = len(dataLoader)

    with torch.no_grad():
        for i, data in enumerate(ds):
            x, y, bboxs = data
            for b in bboxs:
                print(b)
            if i > 20:
                break

## [105, 73, 224, 166]
# [141, 40, 224, 130]
# [29, 48, 224, 138]
# [106, 150, 163, 181]
# [99, 101, 209, 155]
# [67, 34, 224, 132]
# [8, 57, 224, 157]
# [90, 67, 211, 114]
# [4, 55, 224, 178]
# [54, 86, 224, 165]
# [28, 74, 224, 179]
# [34, 55, 177, 161]
# [41, 100, 220, 184]
# [41, 53, 224, 163]
# [22, 25, 224, 135]
# [35, 102, 208, 177]
# [24, 24, 224, 122]
# [2, 35, 224, 129]
# [16, 6, 224, 156]
# [62, 70, 224, 166]
# [109, 111, 224, 159]
# [78, 113, 170, 160]
# [36, 0, 224, 167]

