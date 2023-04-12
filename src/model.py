# Importing Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#Importing DataPreprocessing.py
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

  return model


# Model-2
# Resnet-50

def resnet_50():
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

    # Creating a pretrained ResNet50 model.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 3)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters())

    epochs = 2
    for epoch in range(epochs):

        print('Epoch:', str(epoch + 1) + '/' + str(epochs))
        print('-' * 5)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_images[phase])
            epoch_accuracy = running_corrects.double() / len(data_images[phase])

            print(phase, ' loss:', epoch_loss, 'epoch_accuracy:', epoch_accuracy)

    return model


"""
def confusion_matrix():

    y_actual = []
    y_test = []

    for i in os.listdir("../Datasets/Chest Xray Kaggle/val/NORMAL"):
        img = load_img("../Datasets/Chest Xray Kaggle/val/NORMAL/"+i, target_size=(224,224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        p = model1.predict_classes(img)
        y_test.append(p[0,0])
        y_actual.append(1)

    for i in os.listdir("../Datasets/Chest Xray Kaggle/val/PNEUMONIA"):
        img = load_img("../Datasets/Chest Xray Kaggle/val/PNEUMONIA/"+i, target_size=(224,224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        p = model1.predict_classes(img)
        y_test.append(p[0,0])
        y_actual.append(1)

    y_actual = np.array(y_actual)
    y_test = np.array(y_test)

    cm = confusion_matrix(y_actual, y_test)
    sns.heatmap(cm, cmap="plasma", annot=True)
    confusion_matrix()
"""

