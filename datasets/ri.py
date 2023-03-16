import os
import torch
import torchvision

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])


default_transform=torchvision.transforms.Compose(transforms=[
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

default_root='F:/DataSet/imagenet/'
default_relabel_root='F:/DataSet/relabel_imagenet/imagenet_efficientnet_l2_sz475_top5/'

# relabel imagenet
class RI(torchvision.datasets.ImageNet):
    def __init__(self,topk=None,
                 root=default_root,relabel_root=default_relabel_root,
                 transform=default_transform):
        super(RI, self).__init__(root=root,split='train',
                                 transform=transform)
        self.topk=topk
        self.relabel_root = relabel_root

    def __getitem__(self, index):
        path, target = self.samples[index]
        # get x
        sample = self.loader(path)
        sample = self.transform(sample)
        # get y
        path=path.replace('\\', '/')
        r_path = os.path.join(self.relabel_root,
                                  '/'.join(path.split('/')[-2:]).split('.')[0] + '.pt')

        rmk = torch.load(r_path)  # relabel map top k
        rmkv,rmki = rmk[0],rmk[1].long()  # split value, index
        rm = torch.zeros(1000,rmkv.shape[1],rmkv.shape[2])  # un-sparsify to 1000 c
        rm = rm.scatter_(0,rmki,rmkv)

        # target = target
        # target = target[1]
        # target = target.int().flatten().bincount()/target.numel()

        target = rm.sum([1,2])
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
    # imageNet = torchvision.datasets.ImageNet('F:/DataSet/imagenet', split='val', transform=default_transform) # 0.68
    # imageNet = torchvision.datasets.ImageNet('F:/DataSet/imagenet', split='train', transform=default_transform) # 0.7991
    rimgnt = RI(topk=topk)   # 0.81267
    indices = np.random.choice(len(rimgnt), num_samples)
    subSet = TD.Subset(rimgnt, indices)
    dataLoader = TD.DataLoader(subSet, batch_size=32, pin_memory=True, num_workers=4, persistent_workers=True)
    num_samples = len(dataLoader)

    model_name = 'vgg16'
    model = torchvision.models.vgg16(pretrained=True).cuda()

    correct=0
    with torch.no_grad():
        for x,y in tqdm(dataLoader):
            x=x.cuda()
            y=y.cuda()
            y_ = TF.softmax(model(x),1).argmax(1)
            correct+=(y_.eq(y)).count_nonzero().item()
    correct/=len(subSet)
    print(correct)



