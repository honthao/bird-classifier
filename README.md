# Birds Classifier



## Project Walkthrough & Codes

* [Project Video Walkthrough]()
* [Code]() for evaluating the real test dataset and outputing the prediction csv file
* [Code]() for calculating train & test loss (Train/Validation Split)

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
2. Tweak data transformations
3. Tuning hyperparameters

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

<p float="middle">
  <img src="/lr_0.1_graph.PNG" width=50% height=50% title="Learning rate = 0.1" />
  <img src="/lr_0.01_graph.PNG" width=50% height=50% title="Learning rate = 0.01" /> 
  <img src="/lr_0.001_graph.PNG" width=50% height=50% title="Learning rate = 0.001" />
</p>

* All three graphs were trained for 15 epochs and each took approximately 2h15m to complete.
* Here’s a table showing their performance after 15 epochs:
![Learning Rates Rerformance](/Learning_Rates_Performance.PNG)

* The graph with lr = 0.1 shows divergent behaviors for the testing loss and is the least stable with the highest testing loss of 2.899 compared to the other two graphs. This shows that the learning rate is a bit too high causing the gradient descent to make big steps and can overshoot the optimum, seeing how the testing losses bounce around in the graph. 
* The graph with lr = 0.01 has the lowest training loss of 0.070 and lowest testing loss of 1.070. But while the training loss continues to decrease, the testing loss doesn’t show much improvement after some iterations. The increasing gap between training and testing loss shows that the model might be overfit.
* The graph with lr = 0.001 has the highest training loss of 0.976, showing that the training progresses slower compared to the other two due to its small learning rate making small adjustments to the model’s weights. 

## Discussion

In terms of learning rates, the results above show that 0.01 performs the best for 15 epochs with an accuracy of 72%. Even though 0.001 shows the best convergence, the gradient descent is slower and it would take more epochs to reach the optimum/improve its accuracy. Instead of increasing the number of epochs and have longer training runtime (high cost) to improve the accuracy of 0.001, I think it would be better to use regularization techniques like dropout or data augmentation to prevent overfitting when lr = 0.01 and increase its accuracy.

## Reflection

Problems I encounter?
* One of the problems that I encountered was the runtime of my model. Because it took pretty long to run even on GPU, it was time-consuming to experiment with different hyperparameters or just tweak the model a bit.

Next steps I would take?
* If I kept working on this project, I would further tune other hyperparameters that I didn't get to due to time and GPU limits. I would also like to try out different regularization techniques and examine which one has the strongest impact on the model's performance.

How my approach differ from others? Was that beneficial?
* I don't know how others approach this problem, but I think my approach probably differs because besides from tuning the hyperparameters, I also mess around with the transformations when loading images. And this has a huge improvement on the model's accuracy as mentioned above.
