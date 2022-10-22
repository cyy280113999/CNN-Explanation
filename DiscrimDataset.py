import torch.utils.data as TD
import torchvision
from PIL import Image

default_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class DiscrimDataset(TD.Dataset):
    def __init__(self, img_dir,image_list,transform=None):
        # 拆开
        self.img_dir=img_dir
        self.ds=[]
        for image_name, clss in image_list:
            for cls in clss:
                self.ds.append([image_name,cls])
        if transform:
            self.transform=transform
        else:
            self.transform=default_transform

    def __getitem__(self, item):
        img_n,cls=self.ds[item]
        img=Image.open(self.img_dir+img_n).convert('RGB')
        if self.transform:
            img=self.transform(img)
        return img,cls

    def __len__(self):
        return len(self.ds)


discrim_hoom='F:/DataSet/discrim_ds/images/'
discrim_imglist = [
['cat_dog_243_282.png',         [243, 282]],
['castle_483_919_970.jpg',      [483, 919, 970]],
['ze1_340_386.jpg', [386, 340]],
['ze2.jpg',         [340, 386]],
['ze3.jpg',         [340, 386]],
['ze4.jpg',         [340, 386]],
['ILSVRC2012_val_00000057.JPEG', [259, 154]],
['ILSVRC2012_val_00000073.JPEG', [21, 22]],
['ILSVRC2012_val_00000476.JPEG', [137, 18]],
['ILSVRC2012_val_00002138.JPEG',[0,758]],
['ILSVRC2012_val_00009111.JPEG',[0,758]],
['ILSVRC2012_val_00037375.JPEG',[0,758]],
['ILSVRC2012_val_00037596.JPEG',[0,758]],
# 150
['',[0,758]],
['',[0,758]],
['',[0,758]],
['',[0,758]],

]


if __name__ == '__main__':

    transformer = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = DiscrimDataset(discrim_hoom, discrim_imglist, transformer)
    dataLoader = TD.DataLoader(ds, batch_size=1, pin_memory=True, num_workers=0)