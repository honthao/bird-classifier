# Birds Classifier



## Problem Description

Train a network model to classify/predict bird species for [the bird classification Kaggle challenge](https://www.kaggle.com/competitions/birds23sp)

## Datasets

I used the Kaggle datasets provided on the challenge website, which include 38562 images (555 classes) for training and 10000 images for testing.

## Pre-existing Work

This project builds upon the [iPynb - Transfer Learning to Birds](https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav?usp=sharing) class tutorial and uses PyTorch framework. I used [ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) as a pretrained model.

## Approach

Using the class tutorial mentioned above as a starting point:
1. Change the pretrained model
    * The model used in the tutorial was ResNet18, achieving _% of accuracy
    * The model I used is ResNet50, achieving _% of accuracy
2. Tweak data transformations before training
3. Experiment with different number of epochs and learning rates

### Data Preparation

I worked in the Kaggle Notebook environment, so I modified the code to reflect this (i.e. where I load in the data).

Changes to data transformations:
* Normalized the images with ImageNet mean and standard deviation. This helps increasing the accuracy to 73.1%
* Resize the images to 224x244. This helps increase the accuracy to 81.2%

```python
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

TRAIN_PATH = '/kaggle/input/birds23sp/birds/train'
TEST_PATH = '/kaggle/input/birds23sp/birds/test'
CHECKPOINTS = '/kaggle/working/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=8, padding_mode='edge'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

trainset = torchvision.datasets.ImageFolder(root=TRAIN_PATH, transform=train_transforms)
testset = torchvision.datasets.ImageFolder(root=TEST_PATH, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

classes = open("/kaggle/input/birds23sp/birds/names.txt").read().strip().split("\n")
class_to_idx = trainset.class_to_idx
idx_to_class = {int(v): int(k) for k, v in class_to_idx.items()}
idx_to_name = {k: classes[v] for k,v in idx_to_class.items()}
```

### Train and Predict Functions

### Load Pretrained Model

### Experiments

## Results

Results here.....
![Graph](/graph.PNG)
<img src="/graph.PNG" width=50% height=50%>

## Discussion

What problems did you encounter?
* One of the problems that I encountered was the runtime of my model. Because it took pretty long to run even on GPU (~2-3h/run), it was time-consuming to experiment with different hyperparameters or just tweak the model a bit.

Are there next steps you would take if you kept working on the project?

How does your approach differ from others? Was that beneficial?
