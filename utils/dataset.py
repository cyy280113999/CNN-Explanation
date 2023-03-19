import torchvision

imageNetDefaultDir = r'F:/DataSet/imagenet/'
imageNetSplits = {
    'train': 'train/',
    'val': 'val/',
}

default_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def loadImageNetClasses(path=imageNetDefaultDir):
    import json
    filename = path + 'imagenet_class_index.json'
    with open(filename) as f:
        c = json.load(f)
        c = {int(i): v[-1] for i,v in c.items()}
        return c


def getImageNetVal(transform=default_transform):
    return torchvision.datasets.ImageNet(root=imageNetDefaultDir,
                                         split="val",
                                         transform=transform)
