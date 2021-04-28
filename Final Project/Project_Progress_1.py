# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 22:06:38 2021

@author: tanjib
"""


import numpy as np 
from tqdm import tqdm
import cv2
import os
import shutil
import itertools
import imutils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import plotly.plotly  as py
from plotly.plotly import iplot
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools
from PIL import Image
from sklearn.preprocessing import OneHotEncoder 

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model,Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping



def load_data(dir_path, img_size=(100,100)):
    """
    Load images as np.arrays to workspace
    """
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                    img = cv2.imread(dir_path + path + '/' + file)
                    X.append(img)
                    y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.figure(figsize = (6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    cm = np.round(cm,2)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def plot_samples(X, y, labels_dict, n=50):
    """
    Creates a gridplot for desired number of images (n) from the specified set
    """
    for index in range(len(labels_dict)):
        imgs = X[np.argwhere(y == index)][:n]
        j = 10
        i = int(n/j)

        plt.figure(figsize=(15,6))
        c = 1
        for img in imgs:
            plt.subplot(i,j,c)
            plt.imshow(img[0])

            plt.xticks([])
            plt.yticks([])
            c += 1
        plt.suptitle('Tumor: {}'.format(labels_dict[index]))
        plt.show()

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)    


def save_new_images(x_set, y_set, folder_name):
    i = 0
    for (img, imclass) in zip(x_set, y_set):
        if imclass == 0:
            cv2.imwrite(folder_name+'NO/'+str(i)+'.jpg', img)
        else:
            cv2.imwrite(folder_name+'YES/'+str(i)+'.jpg', img)
        i += 1
        

def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_new)
        

RANDOM_SEED = 123


IMG_PATH = 'C:\\Users\\tanji\\Desktop\\MachineLearning\\archive\\brain_tumor_dataset\\'
dest='C:\\Users\\tanji\\Desktop\\MachineLearning\\archive\\Processed\\'

## Making Directory
os.mkdir(dest+'TRAIN')
os.mkdir(dest+'TRAIN\\YES')
os.mkdir(dest+'TRAIN\\NO')

os.mkdir(dest+'VAL')
os.mkdir(dest+'VAL\\YES')
os.mkdir(dest+'VAL\\NO')

os.mkdir(dest+'TEST')
os.mkdir(dest+'TEST\\YES')
os.mkdir(dest+'TEST\\NO')

# split the data by train/val/test
for CLASS in os.listdir(IMG_PATH):    
    if not CLASS.startswith('.'):
        IMG_NUM = len(os.listdir(IMG_PATH + CLASS))
        for (n, FILE_NAME) in enumerate(os.listdir(IMG_PATH + CLASS)):
            img = IMG_PATH + CLASS + '\\' + FILE_NAME
            if n < 5:
                shutil.copy(img, dest+'TEST\\' + CLASS.upper() + '\\' + FILE_NAME)
            elif n < 0.8*IMG_NUM:
                shutil.copy(img, dest+'TRAIN\\'+ CLASS.upper() + '\\' + FILE_NAME)
            else:
                shutil.copy(img, dest+'VAL\\'+ CLASS.upper() + '\\' + FILE_NAME)
                
                
TRAIN_DIR = dest+'TRAIN\\'
TEST_DIR =  dest+'TEST\\'
VAL_DIR =  dest+'VAL\\'
IMG_SIZE = (224,224) #Size of VGG-16



# load the image data into workspace
X_train, y_train, labels = load_data(TRAIN_DIR, IMG_SIZE)
X_test, y_test, _ = load_data(TEST_DIR, IMG_SIZE)
X_val, y_val, _ = load_data(VAL_DIR, IMG_SIZE)

plot_samples(X_train, y_train, labels, 30)


## Histogram of data

y = dict()
y[0] = []
y[1] = []
for set_name in (y_train, y_val, y_test):
    y[0].append(np.sum(set_name == 0))
    y[1].append(np.sum(set_name == 1))

data = [y[0],y[1]]
X_bar = np.arange(3)
fig, ax = plt.subplots()
# ax = fig.add_axes([0,0,1,1])
ax.bar(X_bar + 0.00, data[0], color = 'b', width = 0.25)
ax.bar(X_bar + 0.25, data[1], color = 'g', width = 0.25)

ax.set_ylabel('Number of Images')
ax.set_title('Data Splitting Outcome')
plt.xticks(X_bar+0.125, ( 'TRAIN', 'VAL', 'TEST'))
ax.legend(labels=['Healthy', 'Tumor'])
plt.show()


## Histogram of image width

RATIO_LIST = []
for set in (X_train, X_test, X_val):
    for img in set:
        RATIO_LIST.append(img.shape[1]/img.shape[0])
        
plt.hist(RATIO_LIST,color='magenta')
plt.title('Image Ratios')
plt.xlabel('Ratio')
plt.ylabel('Number of Images')
plt.show()


## Cropping each image
# X_train_crop = crop_imgs(set_name=X_train)
# X_val_crop = crop_imgs(set_name=X_val)
# X_test_crop = crop_imgs(set_name=X_test)

# plot_samples(X_train_crop, y_train, labels, 30)

## Cropping directory
# TC=TRAIN_DIR+'TRAIN_CROP'
# TC_Y=TC+'\\YES'
# TC_N=TC+'\\NO'
# os.mkdir(TC, mode = 0o777)
# os.mkdir(TC_Y, mode = 0o777)
# os.mkdir(TC_N, mode = 0o777)

# ValC=VAL_DIR+'VAL_CROP'
# ValC_Y=ValC+'\\YES'
# ValC_N=ValC+'\\NO'
# os.mkdir(ValC, mode = 0o777)
# os.mkdir(ValC_Y, mode = 0o777)
# os.mkdir(ValC_N, mode = 0o777)

# TestC=TEST_DIR+'TEST_CROP'
# TestC_Y=TestC+'\\YES'
# TestC_N=TestC+'\\NO'
# os.mkdir(TestC, mode = 0o777)
# os.mkdir(TestC_Y, mode = 0o777)
# os.mkdir(TestC_N, mode = 0o777)

# save_new_images(X_train_crop, y_train, folder_name=TC+'\\')
# save_new_images(X_val_crop, y_val, folder_name=ValC+'\\')
# save_new_images(X_test_crop, y_test, folder_name=TestC+'\\')


## Preprocessing data to input VGG-16 format
# X_train_prep = preprocess_imgs(set_name=X_train_crop, img_size=IMG_SIZE)
# X_test_prep = preprocess_imgs(set_name=X_test_crop, img_size=IMG_SIZE)
# X_val_prep = preprocess_imgs(set_name=X_val_crop, img_size=IMG_SIZE)

X_train_prep = preprocess_imgs(set_name=X_train, img_size=IMG_SIZE)
X_test_prep = preprocess_imgs(set_name=X_test, img_size=IMG_SIZE)
X_val_prep = preprocess_imgs(set_name=X_val, img_size=IMG_SIZE)


## Loading the model
base_model = VGG16(
    weights='imagenet',
    include_top=False, 
    input_shape=IMG_SIZE + (3,)
)

NUM_CLASSES = 1

model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

model.layers[0].trainable = False

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=1e-4),
    metrics=['accuracy']
)

model.summary()

## Training the model
history = model.fit(X_train_prep, y_train, epochs = 30, batch_size = 60, verbose = 1,validation_data = (X_val_prep, y_val))


## Plotting Accuracy loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(history.epoch) + 1)

plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Set')
plt.plot(epochs_range, val_acc, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Set')
plt.plot(epochs_range, val_loss, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')

plt.tight_layout()
plt.show()


# validate on val set
predictions = model.predict(X_val_prep)
predictions = [1 if x>0.5 else 0 for x in predictions]

accuracy = accuracy_score(y_val, predictions)
print('Val Accuracy = %.2f' % accuracy)

confusion_mtx = confusion_matrix(y_val, predictions) 
cm = plot_confusion_matrix(confusion_mtx, classes = list(labels.items()), normalize=False)

# validate on test set
predictions = model.predict(X_test_prep)
predictions = [1 if x>0.5 else 0 for x in predictions]

accuracy = accuracy_score(y_test, predictions)
print('Test Accuracy = %.2f' % accuracy)

confusion_mtx = confusion_matrix(y_test, predictions) 
cm = plot_confusion_matrix(confusion_mtx, classes = list(labels.items()), normalize=False)



##Cropping Demonstration
img = cv2.imread('C:\\Users\\tanji\\Desktop\\MachineLearning\\archive\\brain_tumor_dataset\\yes\\Y108.jpg')
img = cv2.imread('C:\\Users\\tanji\\Desktop\\MachineLearning\\archive\\brain_tumor_dataset\\yes\\Y29.jpg')
img = cv2.imread('C:\\Users\\tanji\\Desktop\\MachineLearning\\archive\\brain_tumor_dataset\\yes\\Y58.jpg')
img = cv2.imread('C:\\Users\\tanji\\Desktop\\MachineLearning\\archive\\brain_tumor_dataset\\no\\no 2.jpg')

img = cv2.resize(
            img,
            dsize=IMG_SIZE,
            interpolation=cv2.INTER_CUBIC
        )
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# find contours in thresholded image, then grab the largest one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)

# find the extreme points
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

# add contour on the image
img_cnt = cv2.drawContours(img.copy(), [c], -1, (0,100,0), 4)

# add extreme points
img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1)
img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1)
img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1)

# crop
ADD_PIXELS = 0
new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

plt.figure(figsize=(15,6))
plt.subplot(141)
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.title('Original Image')
plt.subplot(142)
plt.imshow(img_cnt)
plt.xticks([])
plt.yticks([])
plt.title('Finding the biggest contour')
plt.subplot(143)
plt.imshow(img_pnt)
plt.xticks([])
plt.yticks([])
plt.title('Determining the extreme points')
plt.subplot(144)
plt.imshow(new_img)
plt.xticks([])
plt.yticks([])
plt.title('Cropping the image')
plt.show()






#Data Aug Demo

demo_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1./255,
    shear_range=0.05,
    brightness_range=[0.1, 1.5],
    horizontal_flip=True,
    vertical_flip=True
)

DataAug_demo=dest+'\\DataAug_demo'
os.mkdir(DataAug_demo, mode = 0o777)
X_train_crop = crop_imgs(set_name=X_train)
x = X_train_crop[20]  
x = x.reshape((1,) + x.shape) 

i = 0
for batch in demo_datagen.flow(x, batch_size=1, save_to_dir=DataAug_demo, save_prefix='aug_img', save_format='jpg'):
    i += 1
    if i > 20:
        break 

plt.imshow(X_train_crop[20])
plt.xticks([])
plt.yticks([])
plt.title('Original Image')
plt.show()

plt.figure(figsize=(15,6))
i = 1
for img in os.listdir(DataAug_demo+'\\'):
    img = cv2.cv2.imread(DataAug_demo + '\\'+ img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(3,7,i)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    i += 1
    if i > 3*7:
        break
plt.suptitle('Augemented Images')
plt.show()
