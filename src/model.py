import pandas as pd
import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import os
import cv2
import glob
import torch
import shutil
import itertools
import torch.nn as nn
import torch.optim as optim
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pathlib import Path
from torch.nn import functional as F
from torchvision import datasets, models, transforms
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from src.DataPreprocessing import resized_images, labels

# Model_1 (Simple Sequential Model)
def model_Sequential():
  X = np.array(resized_images) / 255.0
  y = np.array(labels)

  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=42)
  model = Sequential([
      Conv2D(32, 3, input_shape=(224, 224, 3), activation='relu'),
      MaxPooling2D(),
      Conv2D(16, 3, activation='relu'),
      MaxPooling2D(),
      Conv2D(16, 3, activation='relu'),
      MaxPooling2D(),
      Flatten(),
      Dense(512, activation='relu'),
      Dense(256, activation='relu'),
      Dense(1, activation='sigmoid')
  ])
  model.summary()
  model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
  model.fit(X_train, y_train,batch_size=32,epochs=4,validation_data=(X_test, y_test))
  plt.plot(model.history.history['accuracy'], label = 'train accuracy')
  plt.plot(model.history.history['val_accuracy'],label = 'test_accuracy')
  plt.legend()
  plt.show()
  plt.plot(model.history.history['loss'], label = 'train loss')
  plt.plot(model.history.history['val_loss'],label = 'test_loss')
  plt.legend()
  plt.show()


# Model-2
# Resnet-50
# Creating the model
# Training our model.
# Finally, we train our model. As you can see below, the lines of code are of a great multitude and complexity.
# The result, however, is a trained model which produces a pleasing accuracy.

def trained_model():
    normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            normalizer
        ]),

        'validation': transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
            normalizer
        ])
    }
    # Next, we use those transforms that we created and apply them to our data, storing it in a new file location.

    data_images = {
        'train': datasets.ImageFolder('../Datasets/Chest Xray Kaggle/train',
                                      data_transforms['train']),
        'validation': datasets.ImageFolder(
            '../Datasets/Chest Xray Kaggle/chest_xray/test',
            data_transforms['validation'])
    }
    dataloaders = {
        'train': torch.utils.data.DataLoader(data_images['train'], batch_size=32, shuffle=True, num_workers=0),
        'validation': torch.utils.data.DataLoader(data_images['validation'], batch_size=32, shuffle=True, num_workers=0)
    }
    # Creating the model
    # After the data has been transformed, we now create a device which makes the model be able to run on a cpu and cuda core.
    # Depending on which you use.

    # Furthermore, we create a pretrained ResNet50 model.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True)
    # Subsequently, we loop over all the parameters of the model and set an attribute of 'requires_grad' to False.
    # This is important as it means that we don't update certain parts of our classifier, which can save a lot of unnecessary computation
    for param in model.parameters():
        param.requires_grad = False
        # Here we build the simple architecture of the model, being a linear input layer, a relu activation function and a linear output layer.

    model.fc = nn.Sequential(
        nn.Linear(2048, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 3)
    ).to(device)
    # Also very importantly, we create the criterion of CrossEntropyLoss and an optimizer of Adam.

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters())

    epochs = 2
    for epoch in range(epochs):

        print('Epoch:', str(epoch + 1) + '/' + str(epochs))
        print('-' * 5)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # this trains the model
            else:
                model.eval()  # this evaluates the model

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # convert inputs to cpu or cuda
                labels = labels.to(device)  # convert labels to cpu or cuda

                outputs = model(inputs)  # outputs is inputs being fed to the model
                loss = criterion(outputs, labels)  # outputs are fed into the model

                if phase == 'train':
                    optimizer.zero_grad()  # sets gradients to zero
                    loss.backward()  # computes sum of gradients
                    optimizer.step()  # preforms an optimization step

                _, preds = torch.max(outputs, 1)  # max elements of outputs with output dimension of one
                running_loss += loss.item() * inputs.size(0)  # loss multiplied by the first dimension of inputs
                running_corrects += torch.sum(preds == labels.data)  # sum of all the correct predictions

            epoch_loss = running_loss / len(data_images[phase])  # this is the epoch loss
            epoch_accuracy = running_corrects.double() / len(data_images[phase])  # this is the epoch accuracy

            print(phase, ' loss:', epoch_loss, 'epoch_accuracy:', epoch_accuracy)

    return model

# Executing the sequential model.

model_Sequential()

# Executing the resnet_50 model.
model = trained_model()
