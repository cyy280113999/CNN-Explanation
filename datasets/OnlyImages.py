import os
from torchvision.datasets.folder import ImageFolder, default_loader

def find_classes(directory):
    classes = (0,)
    class_to_idx = {"":0}
    return classes, class_to_idx

class OnlyImages(ImageFolder):
    def find_classes(self, directory: str):
        return find_classes(directory)