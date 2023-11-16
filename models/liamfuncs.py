import numpy as np
import torch
import os
import sys
from torchvision import transforms, datasets, io
from torchvision.transforms import functional

def F1_score(pred, real):
    """ Calculate confusion matrix, recall, precision and F1 score for
    classification data. Used for MNIST data with pytorch"""
    n = 10
    conf = np.zeros((n, n))

    # Count results to build the confusion matrix
    for i in range(len(pred)):
        conf[pred[i], real[i]] += 1

    if np.count_nonzero(np.identity(10)*conf) < 10: # zero rows will break it!
        conf = conf + np.identity(10)*0.1
        #raise ValueError("No correct classificiations for some values!")
    
    # Precision, Recall, F1
    precision = np.max(np.identity(10)*conf, axis=1)/np.sum(conf, axis=1)
    recall = np.max(np.identity(10)*conf, axis=1)/np.sum(conf, axis=0).T
    F1 = 2*precision*recall/(precision+recall)
    return F1, conf, recall, precision

def data_import(dir, cuda_enable=True, test_only=False):
    # import training data
    imgdir = dir+"\\train"
    transform = transforms.Compose([transforms.ToTensor()])
    train_raw = datasets.ImageFolder(imgdir, transform=transform)
    class_raw = train_raw.find_classes(imgdir)
    image_idx = class_raw[1]['image'] # label index of images
    mask_idx = class_raw[1]['mask'] # label index of masks
    train_mapping = train_raw.make_dataset(imgdir, class_raw[1], ['png'])

    if not test_only:
        x_train = []
        y_train = []
        for imgdir, idx in train_mapping:
            imgtsr = io.read_image(imgdir, mode=io.ImageReadMode.RGB)
            if idx == image_idx:
                x_train.append(imgtsr)
                imgtsr = functional.rotate(imgtsr, angle=90)
                x_train.append(imgtsr)
                imgtsr = functional.rotate(imgtsr, angle=90)
                x_train.append(imgtsr)
                imgtsr = functional.rotate(imgtsr, angle=90)
                x_train.append(imgtsr)
            elif idx == mask_idx:
                y_train.append(imgtsr)
                imgtsr = functional.rotate(imgtsr, angle=90)
                y_train.append(imgtsr)
                imgtsr = functional.rotate(imgtsr, angle=90)
                y_train.append(imgtsr)
                imgtsr = functional.rotate(imgtsr, angle=90)
                y_train.append(imgtsr)
            else:
                raise ValueError("Unknown Label")

        # Normalize data
        # x_train = torch.mean(torch.stack(x_train,dim=0)/255, dim=1, keepdim=True)
        y_train = torch.mean(torch.stack(y_train,dim=0)/255, dim=1, keepdim=False) #(80, 1, 512, 512)
        x_train = torch.stack(x_train,dim=0)/255
        # y_train = torch.stack(y_train,dim=0)/255 #(80, 3, 512, 512)

        x_valid = x_train[-32:]
        y_valid = y_train[-32:]
    else:
        x_test = None
        y_test = None
        x_valid = None
        y_valid = None

    # import test data
    imgdir = dir+"\\test"
    transform = transforms.Compose([transforms.ToTensor()])
    test_raw = datasets.ImageFolder(imgdir, transform=transform)
    test_mapping = test_raw.make_dataset(imgdir, class_raw[1], ['png'])

    x_test = []
    y_test = []
    for imgdir, idx in test_mapping:
        imgtsr = io.read_image(imgdir, mode=io.ImageReadMode.RGB)
        if idx == image_idx:
            x_test.append(imgtsr)
        elif idx == mask_idx:
            y_test.append(imgtsr)
        else:
            raise ValueError("Unknown Label")

    # x_test = torch.mean(torch.stack(x_test,dim=0)/255, dim=1, keepdim=True)
    y_test = torch.mean(torch.stack(y_test,dim=0)/255, dim=1, keepdim=False) #(20, 1, 512, 512)
    x_test = torch.stack(x_test,dim=0)/255
    # y_test = torch.stack(y_test,dim=0)/255 #(20, 3, 512, 512)
    if test_only:
        return x_test, y_test
    else:
        if cuda_enable:
            return x_train[0:-32].cuda(), y_train[0:-32].cuda(), x_valid.cuda(), y_valid.cuda(), x_test.cuda(), y_test.cuda()
        else:
            return x_train[0:-32], y_train[0:-32], x_valid, y_valid, x_test, y_test
        
def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def IoU_loss(input, target):
    mask_pred = torch.round(input).view(-1)
    mask_targ = torch.round(target).view(-1)

    tp = (mask_pred*mask_targ).sum()
    fn = (mask_targ - mask_pred*mask_targ).sum()
    fp = (mask_pred*torch.abs(mask_targ-1)).sum()

    return float(tp / (tp + fn + fp))