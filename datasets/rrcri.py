import os
import torch
import torchvision
from torchvision.transforms.functional import resized_crop
from torchvision.ops import roi_align


default_root='F:/DataSet/imagenet/'
default_relabel_root='F:/DataSet/relabel_imagenet/imagenet_efficientnet_l2_sz475_top5/'

# random resize crop
class RandomResizedCropWithCoords(torchvision.transforms.RandomResizedCrop):
    def __call__(self, img):
        x0, y0, h, w = self.get_params(img, self.scale, self.ratio)
        coords = (x0 / img.size[1], y0 / img.size[0],
                  h / img.size[1],  w / img.size[0])
        coords = torch.tensor(coords)  # given by x0,y0,h,w. by percentage because relabel is low resolution(224->15)
        coords[2:]+=coords[:2]  # given by x0 y0 x1 y1.
        img=resized_crop(img, x0, y0, h, w, self.size,self.interpolation)
        return img, coords
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
class ComposeWithCoords(torchvision.transforms.Compose):
    def __call__(self, img):
        coords = None
        for t in self.transforms:
            if type(t).__name__ == 'RandomResizedCropWithCoords':
                img, coords = t(img)
            elif type(t).__name__ == 'RandomCropWithCoords':
                img, coords = t(img)
            elif type(t).__name__ == 'RandomHorizontalFlipWithCoords':
                img, coords = t(img, coords)
            else:
                img = t(img)
        return img, coords

default_transform=ComposeWithCoords(transforms=[
    RandomResizedCropWithCoords(size=(224,224),scale=(0.5,1),ratio=(1,1)),
    torchvision.transforms.ToTensor(),
    normalize
])


# use random resize crop---relabel imagenet
class RRCRI(torchvision.datasets.ImageNet):
    def __init__(self,topk=None,
                 root=default_root,relabel_root=default_relabel_root,
                 transform=default_transform):
        super(RRCRI, self).__init__(root=root, split='train',
                                    transform=transform)
        self.topk=topk
        self.relabel_root = relabel_root

    def __getitem__(self, index):
        path, target = self.samples[index]
        # get x
        sample = self.loader(path)
        sample,coord = self.transform(sample)
        # get y
        path=path.replace('\\', '/')
        r_path = os.path.join(self.relabel_root,
                                  '/'.join(path.split('/')[-2:]).split('.')[0] + '.pt')

        rmk = torch.load(r_path)  # relabel map top k
        rmkv,rmki = rmk[0],rmk[1].long()  # split value, index
        rm = torch.zeros(1000,rmkv.shape[1],rmkv.shape[2])  # un-sparsify to 1000 c
        rm = rm.scatter_(0,rmki,rmkv)
        coord=coord * rm.shape[1]  # like rm shape
        coord = torch.cat([torch.tensor([0]),coord])  # insert 0 to length 5. require by roi_align [K,5]
        target = roi_align(input=rm.unsqueeze(0),  # avg pooling with region to score
                                   boxes=coord.unsqueeze(0)-0.5,output_size=(1, 1))  # aligned=False, -0.5
        target = target.flatten()
        target = torch.softmax(target, 0)  # prob of 1000 c
        if self.topk is not None:  # return label idx
            target = target.argsort(0,True)[self.topk]
        return sample, target

if __name__ == '__main__':
    import numpy as np
    import torch.nn.functional as TF
    import torch.utils.data as TD
    from tqdm import tqdm
    num_samples = 100000
    topk = 0
    # imageNet = torchvision.datasets.ImageNet('F:/DataSet/imagenet', split='val', transform=transformer)
    rimgnt = RRCRI(topk=topk)
    indices = np.random.choice(len(rimgnt), num_samples)
    subSet = TD.Subset(rimgnt, indices)
    dataLoader = TD.DataLoader(subSet, batch_size=32, pin_memory=True, num_workers=4, persistent_workers=True)
    num_samples = len(dataLoader)

    model_name = 'vgg16'
    model = torchvision.models.vgg16(pretrained=True).cuda()

    correct = 0
    for x, y in tqdm(dataLoader):
        x = x.cuda()
        y = y.cuda()
        y_ = TF.softmax(model(x), 1).argmax(1)
        correct += (y_.eq(y)).count_nonzero().item()
    correct /= len(subSet)
    print(correct)
# 0.79
