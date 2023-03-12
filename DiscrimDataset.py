import torch.utils.data as TD
import torchvision
from PIL import Image


default_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


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
['ILSVRC2012_val_00000476.JPEG', [18, 137]],
['ILSVRC2012_val_00002138.JPEG',[0,758]],
['ILSVRC2012_val_00009111.JPEG',[0,758]],
['ILSVRC2012_val_00037375.JPEG',[0,758]],
['ILSVRC2012_val_00037596.JPEG',[0,758]],
['ILSVRC2012_val_00010074.JPEG',[3,983]],
['ILSVRC2012_val_00018466.JPEG',[3,983]],
['ILSVRC2012_val_00023935.JPEG',[3,983]],
['ILSVRC2012_val_00024854.JPEG',[3,983]],
['ILSVRC2012_val_00025137.JPEG',[3,983]],
['ILSVRC2012_val_00025686.JPEG',[3,983]],
['ILSVRC2012_val_00034413.JPEG',[3,983]],
['ILSVRC2012_val_00037915.JPEG',[3,983]],
['ILSVRC2012_val_00009237.JPEG',[467,282]],
['ILSVRC2012_val_00000911.JPEG',[7,8]],
['ILSVRC2012_val_00004306.JPEG',[7,8]],
['ILSVRC2012_val_00004463.JPEG',[7,84]],
['ILSVRC2012_val_00003279.JPEG',[14.11]],
['ILSVRC2012_val_00043965.JPEG',[14,448]],
['ILSVRC2012_val_00001146.JPEG',[84,8,86]],
['ILSVRC2012_val_00005400.JPEG',[667,400]],
['ILSVRC2012_val_00012083.JPEG',[400,667]],
['ILSVRC2012_val_00020475.JPEG',[400,454,624]],
['ILSVRC2012_val_00029372.JPEG',[400,667]],
['ILSVRC2012_val_00038190.JPEG',[400,667]],
['ILSVRC2012_val_00047254.JPEG',[400,624]],
['ILSVRC2012_val_00047602.JPEG',[400,667]],
['ILSVRC2012_val_00001080.JPEG',[455,586]],
['ILSVRC2012_val_00006337.JPEG',[440,455,737]],#38000,760
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
# ['',[]],
]



class DiscrimDataset(TD.Dataset):
    def __init__(self, img_dir=discrim_hoom, img_list=discrim_imglist, transform=None):
        # 拆开
        self.img_dir=img_dir
        self.ds=[]
        for image_name, clss in img_list:
            for cls in clss:
                self.ds.append([image_name,cls])
        if transform is not None:
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

if __name__ == '__main__':

    transformer = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = DiscrimDataset(discrim_hoom, discrim_imglist, transformer)
    dataLoader = TD.DataLoader(ds, batch_size=1, pin_memory=True, num_workers=0)