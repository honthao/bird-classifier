# Birds Classifier



## Project Walkthrough & Codes

* [Project Video Walkthrough](https://drive.google.com/file/d/1v-8p6JcCCFJAMPZWvyRiJuXC9g1ij4sp/view?usp=sharing)
* [Code 1](https://drive.google.com/file/d/1o99Yy67VaEarS3ZgMxoowgrItTUV0rUs/view?usp=sharing) for evaluating the real test dataset and outputing the prediction csv file (Only the latest experiment output)
* [Code 2](https://drive.google.com/file/d/1JE2sgIn-knOtIPmnqv89gYXso3toyavP/view?usp=sharing) for calculating train & test loss (Train/Validation Split) (Only include the latest experiment output)

## Problem Description

Train a network model to classify/predict bird species to the highest accuracy for [the bird classification Kaggle challenge](https://www.kaggle.com/competitions/birds23sp)

## Datasets

I used the Kaggle datasets provided on the challenge website, which include 38562 images (555 classes) for training and 10000 images for testing. Only training directory has labels, which is in names.txt

## Pre-existing Work

This project builds upon the [iPynb - Transfer Learning to Birds](https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav?usp=sharing) and the [ImageNet and Transfer Learning](https://colab.research.google.com/drive/1EBz4feoaUvz-o_yeMI27LEQBkvrXNc_4?usp=sharing#scrollTo=76z4UJgTR13J) class tutorials and uses PyTorch framework. I used [ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) as a pretrained model.

## Approach

I'll be using Transfer Learning to accomplish this challenge, which involves using a pretrained model that has been trained on the ImageNet dataset and applying it to this bird challenge dataset. Using the class tutorial mentioned above as a starting point:
1. Initialize data and choose a pretrained model
2. Tweak data transformations
3. Split training dataset into train/split and tune hyperparameters

## Code Preparation

### 1.1) Initialize data

I worked in the Kaggle Notebook environment, so I modified the tutorial code to reflect this (i.e. where I load in the data).

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
    # This tutorial code will be change later
    transforms.Resize(128),
    transforms.RandomCrop(128, padding=8, padding_mode='edge'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    # This tutorial code will be change later
    transforms.Resize(128),
    transforms.ToTensor(),
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

### 1.2) Load Pretrained Model

I chose ResNet50 as the starting point for my network model. 

```python
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(idx_to_name))
```

### Train and Predict Methods

I used the same train and predict methods from the tutorial. Here is the code for reference:

```python
def train(model, dataloader, epochs=1, start_epoch=0, lr=0.01, momentum=0.9, decay=0.0005, 
          verbose=1, print_every=10, state=None, schedule={}, checkpoint_path=None):
    model.to(device)
    model.train()
    losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

    # Load previous training state
    if state:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        losses = state['losses']

    # Fast forward lr schedule through already trained epochs
    for epoch in range(start_epoch):
        if epoch in schedule:
            print ("Learning rate: %f"% schedule[epoch])
            for g in optimizer.param_groups:
                g['lr'] = schedule[epoch]

    for epoch in range(start_epoch, epochs):
        sum_loss = 0.0
        
        # Update learning rate when scheduled
        if epoch in schedule:
            print ("Learning rate: %f"% schedule[epoch])
            for g in optimizer.param_groups:
                g['lr'] = schedule[epoch]

        for i, batch in enumerate(dataloader, 0):
            inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            sum_loss += loss.item()

            if i % print_every == print_every-1:  # print every 10 mini-batches
                if verbose:
                    print('[%d, %5d] loss: %.3f' % (epoch, i + 1, sum_loss / print_every))
                sum_loss = 0.0
        if checkpoint_path:
            state = {'epoch': epoch+1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'losses': losses}
            torch.save(state, checkpoint_path + 'checkpoint-%d.pkl'%(epoch+1))
    return losses

def predict(model, dataloader, ofname):
    out = open(ofname, 'w')
    out.write("path,class\n")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader, 0):
            if i%100 == 0:
                print(i)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            fname, _ = dataloader.dataset.samples[i]
            out.write("test/{},{}\n".format(fname.split('/')[-1], idx_to_class[predicted.item()]))
    out.close()
```

## Experiments

Before tweaking and tuning my model, the model I have currently (what specified above) achieves a 70.05% of accuracy with the following command:

```python
train(model, trainloader, epochs=10, lr=.01, print_every=10, checkpoint_path=CHECKPOINTS)
```

Let's start with the changes to see how much my model can improve.

### 2) Data Transformations

##### 2.1) Normalize Images

Normalizing the training and testing images with ImageNet mean and standard deviation. This helps increasing the accuracy to 73.1%

```python
train_transforms = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomCrop(128, padding=8, padding_mode='edge'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
```

##### 2.2) Resize Images

Resize the images to 224x224 instead of 128x128. This helps increasing the accuracy to 81.2%

```python
train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=8, padding_mode='edge'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])  # normalize im w/ ImageNet mean and std
])

test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
```

### 3) Tuning Hyperparameters (Learning Rates & Number of Epochs)

#### 3.1) Code Preparation

Before tuning my hyperparameters, I'll split my training dataset into 80% training and 20% validation/testing to be able to examine train vs test loss. This will be in a separate code file from above (code 1) to avoid messing up the code, and I also link this code as code 2 in the Project Walkthrough & Codes section.

**3.1.1) Loading and Splitting Training Dataset**

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
from torch.utils.data import random_split

TRAIN_PATH = '/kaggle/input/birds23sp/birds/train'
TEST_PATH = '/kaggle/input/birds23sp/birds/test'
CHECKPOINTS = '/kaggle/working/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Split trainset into 80% train & 20% test
trainset = torchvision.datasets.ImageFolder(root=TRAIN_PATH, transform=train_transforms)
train_size = int(0.8 * len(trainset))
train_ds, test_ds = random_split(trainset, [train_size, len(trainset)-train_size])
trainloader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

classes = open("/kaggle/input/birds23sp/birds/names.txt").read().strip().split("\n")
class_to_idx = trainset.class_to_idx
idx_to_class = {int(v): int(k) for k, v in class_to_idx.items()}
idx_to_name = {k: classes[v] for k,v in idx_to_class.items()}
```

**3.1.2) Modifying Training and Predict Methods**
    
The training method is pretty much the same as the tutorial except I added in some lines and modified the function to also calculate and return the testing/validation loss.

```python
def train(model, trainload, testload, epochs=1, start_epoch=0, lr=0.01,
        momentum=0.9, decay=0.0005, verbose=1, state=None, checkpoint_path=None):
model.to(device)
train_losses = []
test_losses = []
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

# Load previous training state
if state:
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
    train_losses = state['losses']

for epoch in range(start_epoch, epochs):
    print("Training...")
    train_sum_loss = 0.0
    model.train()
    for i, batch in enumerate(trainload, 0):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_sum_loss += loss.item()
    train_losses.append(train_sum_loss / len(trainload))
    if verbose:
        print('%d, loss: %.3f' % (epoch, train_sum_loss / len(trainload)))

    if checkpoint_path:
        state = {'epoch': epoch+1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'losses': train_losses}
        torch.save(state, checkpoint_path + 'checkpoint-%d.pkl'%(epoch+1))
    
    print("Testing...")
    test_sum_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(testload, 0):
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_sum_loss += loss.item()
        test_losses.append(test_sum_loss / len(testload))
        if verbose:
            print('%d, loss: %.3f' % (epoch, test_sum_loss / len(testload)))

return (train_losses,test_losses)
``` 

I used the accuracy method from the "ImageNet and Transfer Learning" class tutorial as my predict method.

```python
def predict(model, dataloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader, 0):
            if i%100 == 0:
                print(i)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network: {100 * correct // total} %')
```

Everything else will be kept the same as the code for evaluating the real test dataset.

#### 3.2) Actual Experiment

I ran the following command three times, one for each learning rates (0.1, 0.01, and 0.001) to see which one yields the best accuracy when train for 15 epochs:

```python
(train_losses, test_losses) = train(model, trainload=trainloader, testload=testloader, epochs=15, lr=.1, checkpoint_path=CHECKPOINTS)
```

Then plotted each one:

```python
epochs = range(1, 16)
plt.plot(epochs, train_losses,'g-')
plt.plot(epochs, test_losses,'r-')
```

| LR = 0.1                  | LR = 0.01                   | LR = 0.001                    |
| ------------------------- | --------------------------- | ----------------------------- |
| ![0.1](/lr_0.1_graph.PNG) | ![0.01](/lr_0.01_graph.PNG) | ![0.001](/lr_0.001_graph.PNG) |

* All three graphs were trained for 15 epochs and each took approximately 2h15m to complete.
* Here’s a table showing their performance after 15 epochs:

![Learning Rates Rerformance](/Learning_Rates_Performance.PNG)

* The graph with lr = 0.1 shows divergent behaviors for the testing loss and is the least stable with the highest testing loss of 2.899 compared to the other two graphs. This shows that the learning rate is a bit too high causing the gradient descent to make big steps and can overshoot the optimum, seeing how the testing losses bounce around in the graph. 
* The graph with lr = 0.01 has the lowest training loss of 0.070 and lowest testing loss of 1.070. But while the training loss continues to decrease, the testing loss doesn’t show much improvement after some iterations. The increasing gap between training and testing loss shows that the model might be overfit.
* The graph with lr = 0.001 has the highest training loss of 0.976, showing that the training progresses slower compared to the other two due to its small learning rate making small adjustments to the model’s weights.

I started out with lr = 0.01 before this experiment with an accuracy of 81.2% (from data transformations), so I already used the best learning rate in this case. But since I only train the model for 10 epochs before while this experiment train for 15 epochs, I ran my code 1 again but with 15 epochs and get an 82.85% accuracy after submitting it.

## Results

Applying different data transformations when processing the images helps increase the accuracy by aprroximately 10% (from 70.05% to 81.2%). I was suprised at how much improvement to the model's accuracy achieves by applying different data transformations alone. But this makes sense because resizing and normalizing the images better fit them with the inputs expected by the pre-trained model. In terms of learning rates, the results above show that 0.01 performs the best for 15 epochs with an accuracy of 72%. Even though 0.001 shows the best convergence, the gradient descent is slower and it would take more epochs than 15 to reach the optimum/improve its accuracy. Instead of increasing the number of epochs and have longer training runtime (high cost) to improve the accuracy of 0.001, I think it would be better to use regularization techniques like dropout or data augmentation to prevent overfitting when lr = 0.01 and increase its accuracy. Starting out with an accuracy of 70.05%, I was able to improve my model to 82.85% and currently place 7th on the leaderboard through applying different data transformations before training and tuning hyperparameters.

### System Demo

Here are some examples of my model predicting the bird species:

* First batch --> 4/5 = 80% correct

![1st Batch](/1st_batch.PNG)

* Second batch --> 4/5 = 80% correct

![2nd Batch](/2nd_batch.PNG)

* Third batch --> 5/5 = 100% correct

![3rd Batch](/3rd_batch.PNG)

## Discussion

Problems I encounter?
* One of the problems that I encountered was the runtime of my model. Because it took pretty long to run even on GPU, it was time-consuming to experiment with different hyperparameters or just tweak the model a bit. So I wasn't able to experiment with more hyperparameters as I had originally planned.
* Another problem was splitting the training directory into train and test because since they both use the same train dataset, I can't introduce randomness when applying transformations to the images in [code 2](#project-walkthrough-&-codes) as I've done in [code 1](#project-walkthrough-&-codes) since we don't want randomness in testing.

Next steps I would take?
* If I kept working on this project, I would further tune other hyperparameters that I didn't get to due to time and GPU limits. I would also like to try out different regularization techniques and examine which one has the strongest impact on the model's performance.

How my approach differ from others? Was that beneficial?
* I don't know how others approach this problem, but I think my approach probably differs because besides from tuning the hyperparameters, I also mess around with the transformations when loading images. And this has a huge improvement on the model's accuracy as mentioned above.
