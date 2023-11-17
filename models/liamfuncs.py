import numpy as np
import torch
import os
import sys
from torchvision import transforms, datasets, io
from torchvision.transforms import functional

def data_import(dir, cuda_enable=True, test_only=False):
    # import training data
    transform = transforms.Compose([transforms.ToTensor()])
    train_raw = datasets.ImageFolder(dir, transform=transform)
    class_raw = train_raw.find_classes(dir)
    image_idx = class_raw[1]['dataset'] # label index of images
    train_mapping = train_raw.make_dataset(dir, class_raw[1], ['png'])

    x_train = []
    for imgdir, idx in train_mapping:
        imgtsr = io.read_image(imgdir, mode=io.ImageReadMode.RGB)
        if idx == image_idx:
            x_train.append(imgtsr)
        else:
            raise ValueError("Unknown Label")

    # Normalize data
    x_train = (torch.stack(x_train,dim=0)/255).cuda()
    return x_train
        
if __name__ == "__main__":
    dirname = "C:\\Users\\Slaye\\Desktop\\repo\\Fall2023_DeepLearnProject\\cache"
    x_train = data_import(dirname)

    print(x_train.size())