import torch
import torchvision
from PIL import Image

device = 'cuda'

def pilOpen(filename):
    return Image.open(filename).convert('RGB')

pilToTensor = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor()
])

pilToRRCTensor = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224, scale=(0.25, 1), ratio=(1, 1)),
    torchvision.transforms.ToTensor()
])

ImgntMean = [0.485, 0.456, 0.406]
ImgntStd = [0.229, 0.224, 0.225]
toStd = torchvision.transforms.Normalize(ImgntMean, ImgntStd)
ImgntMeanTensor = torch.tensor(ImgntMean).reshape(1, -1, 1, 1)
ImgntStdTensor = torch.tensor(ImgntStd).reshape(1, -1, 1, 1)


def invStd(tensor):
    tensor = tensor * ImgntStdTensor + ImgntMeanTensor
    return tensor


def get_image_x(filename='cat_dog_243_282.png', image_folder='input_images/', device=device):
    # require folder , pure filename
    filename = image_folder + filename
    img_PIL = pilOpen(filename)
    img_tensor = pilToTensor(img_PIL).unsqueeze(0)
    img_tensor = toStd(img_tensor).to(device)
    return img_tensor