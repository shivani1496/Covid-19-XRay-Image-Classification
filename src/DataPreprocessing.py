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

# Global Variables
covid_images = []
normal_images = []
resized_images = []
labels = []

# Path Variables for Covid Data
FILE_PATH = "../Datasets/covid-chestxray-dataset-master/metadata.csv"
IMAGE_PATH = "../Datasets/covid-chestxray-dataset-master/images"
TARGET_COVID_DIR = "../Datasets/Covid19"

# Path Variables for non-covid data
KAGGLE_FILE_PATH = "../Datasets/Chest Xray Kaggle/train/NORMAL"
TARGET_NORMAL_DIR = "../Datasets/Normal"

# Path to main dir
main_path = "../Datasets/"

def extract_covid_images(FILE_PATH, IMAGE_PATH, TARGET_COVID_DIR):

    df = pd.read_csv(FILE_PATH)
    print(df.shape)
    df.head(30)

    if not os.path.exists(TARGET_COVID_DIR):
        os.mkdir(TARGET_COVID_DIR)
        print("Covid19 Folder created")

    covid_positive = 0
    for (i, row) in df.iterrows():
        if row["finding"] == "Pneumonia/Viral/COVID-19" and row["view"] == "PA":
            filename = row["filename"]
            # extracting images from this path
            image_path = os.path.join(IMAGE_PATH, filename)
            # pasting the extracted images into this new path
            image_copy_path = os.path.join(TARGET_COVID_DIR, filename)
            shutil.copy2(image_path, image_copy_path)
            print("Extracting Images...", covid_positive)
            covid_positive += 1

    print(covid_positive)

def extract_normal_images(KAGGLE_FILE_PATH, TARGET_NORMAL_DIR):

    if not os.path.exists(TARGET_NORMAL_DIR):
        os.mkdir(TARGET_NORMAL_DIR)
        print("Normal Folder created")

    # randomly pick 196 normal images
    image_names = os.listdir(KAGGLE_FILE_PATH)
    random.shuffle(image_names)

    for i in range(196):
        image_name = image_names[i]
        # source image path
        image_path = os.path.join(KAGGLE_FILE_PATH, image_name)
        target_path = os.path.join(TARGET_NORMAL_DIR, image_name)
        # copying images from source to target
        shutil.copy2(image_path, target_path)
        print("copying image", i)

# Steps for Data PreProcessing begins....
def cnvt_img2arr(covid_dir, normal_dir):

    for filename in os.listdir(covid_dir):
        img = cv2.imread(os.path.join(covid_dir, filename))
        if img is not None:
            covid_images.append(img)

    for filename in os.listdir(normal_dir):
        img = cv2.imread(os.path.join(normal_dir, filename))
        if img is not None:
            normal_images.append(img)

    print(covid_images)
    print(normal_images)

# Converting Images to RGB, Resizing the images and Label Creation

def cnvt_img2RGB():
    img_size = 224  # used for VGG-16 CNN model

    for image in covid_images:
        resized_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image, (img_size, img_size))
        resized_images.append(resized_image)
        labels.append(1)  # 1 for covid19

    for image in normal_images:
        resized_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image, (img_size, img_size))
        resized_images.append(resized_image)
        labels.append(0)  # 0 for normal(non-covid)

    print(labels)

def plot_xray_images(main_path, i):

    covid_img = os.listdir(main_path+"Covid19")
    nrml_img = os.listdir(main_path+"Normal")

    normal = cv2.imread(main_path+"Normal//"+nrml_img[i])
    normal = skimage.transform.resize(normal, (150, 150, 3))
    covid = cv2.imread(main_path+"Covid19//"+covid_img[i])
    covid = skimage.transform.resize(covid, (150, 150, 3))
    pair = np.concatenate((normal, covid), axis=1)
    print(" Normal Chest X-Ray v/s Covid19 Chest X-Ray")
    plt.figure(figsize=(10, 5))
    plt.imshow(pair)
    plt.show


def train_test_split():
    # Converting Data to numpy arrays while scaling the pixel intensities to the range 0, 1
    X = np.array(resized_images) / 255.0
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=42)


def data_pre_processing():

    print("Step 1: Coverting Images to Array...")
    cnvt_img2arr(TARGET_COVID_DIR, TARGET_NORMAL_DIR)
    print("Step 2: Converting Images to RGB, Resizing the images and Label Creation")
    cnvt_img2RGB()
    print("Step 3: Visulaization - Plotting X-Ray Images ")
    for i in range(0,5):
        plot_xray_images(main_path, i)
    print("Step 4: Splitting into train and test")
    train_test_split()

extract_covid_images(FILE_PATH, IMAGE_PATH, TARGET_COVID_DIR)
extract_normal_images(KAGGLE_FILE_PATH, TARGET_NORMAL_DIR)
data_pre_processing()



