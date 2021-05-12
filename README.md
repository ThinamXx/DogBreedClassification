# **IMAGE CLASSIFICATION: DOG BREED CLASSIFICATION**

**OBJECT DETECTION**
- Object Recognition is a computer technology related to computer vision and image processing that deals with detecting instances of semantic objects of a certain class in digital images and videos. It has applications in many areas of computer vision, including image retrieval and video surveillance.

**LIBRARIES AND DEPENDENCIES**
- I have listed all the necessary libraries and dependencies required for this project here:

```python
import os, collections, math
import shutil
import pandas as pd

import torch
import torchvision
from torch import nn
from d2l import torch as d2l
```

**OBTAINING AND ORGANIZING THE DATASET**
- I have used google colab for this project so the process of downloading and reading the data might be different in other platforms. I will use [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification) for this project. The dataset is divided into training set and test set. There are 120 breeds of dogs in the training dataset including Labradors, Poodles, Dachshunds, Samoyeds, Huskies, Chihuahuas and Yorkshire Terriers.

**NEURAL NETWORKS MODEL**
- I will use image augmentation to cope with overfitting. The images are flipped at random and normalized. I will use the pretrained ResNet34 model. I will use the input of the pretrained model output layer which is the extracted features. I will replace the output layer with a small custom output layer that can be trained. I will first use the member variable features to obtain the input of the pretrained model output layer which is the extracted features. Then I will use the features as the input for our small custom output network and compute the output. I have presented the implementation of Image Augmentation and Normalization, Defining Neural Networks Model and Loss Function using PyTorch here in the Snapshot.

![IMAGE](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20160.PNG)

**TRAINING THE MODEL**
- I will define model training function train here. I will select the model and tune hyperparameters according to the model performance on the validation set. The model training function train only trains the small custom output network. I will train and validate the model. I have presented the implementation of Defining the Training Function using PyTorch here in the snapshot. 

![IMAGE](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20161.PNG)
