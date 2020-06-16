# -*- encoding: utf-8 -*-
'''
@File    :   ROC.py
@Time    :   2020/05/09 17:24:38
@Author  :   yetaoyu 
@Version :   1.0
@Contact :   taoyuye@gmail.com
@Desc    :   Decide what model to use by modifying the variable base_model="?"
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
start_time = time.time()

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import keras
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Model, load_model
from keras.applications import VGG16, InceptionV3, InceptionResNetV2, ResNet50
import numpy as np
import os
import cv2
import tensorflow as tf
from os import scandir
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

from keras.wrappers.scikit_learn import KerasClassifier
import itertools
import sys
import math 
import random
import string
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from itertools import cycle
from scipy import interp

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

matplotlib.rcParams.update({'font.size': 11}) 

def load_data(input_dir, image_width, image_height): # input_dir = train
    file_paths = []
    data = []
    labels = []
    class_names = []
    label = -1

    for img_fold in scandir(input_dir): # img_fold = ACC
        class_names.append(img_fold.path.split("/")[-1])
        # print("img_fold=",class_names)
        # print("cancer_path=",img_fold.path)
        label = label+1
        for img_file in scandir(img_fold.path):
            # print("sample_path=", img_file.path)
            file_paths.append(img_file.path)
            image = cv2.imread(img_file.path)
            image = cv2.resize(image,(image_width, image_height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data.append(image)
            labels.append(label)
    # print("class_names=", class_names)
    # print("labels=", labels)
    return np.array(data), np.array(labels), class_names

IMAGE_SIZE = 310
num_classes = 36
input_shape = (IMAGE_SIZE,IMAGE_SIZE,3)

path = os.getcwd() + '/dataset'
train_path = path + '/train'
valid_path = path + '/valid'
test_path = path + '/test'
print("train_path = ", train_path)

X_train, y_train, class_names = load_data(train_path, IMAGE_SIZE, IMAGE_SIZE)
X_valid, y_valid, class_names2 = load_data(valid_path, IMAGE_SIZE, IMAGE_SIZE)
X_test, y_test, class_names3 = load_data(test_path, IMAGE_SIZE, IMAGE_SIZE)
print("class_names = ", class_names)
print("class_names2 = ", class_names2)
print("class_names3 = ", class_names3)
input_shape = X_train.shape[1:]

# Normalize data.
X_train = X_train.astype('float32') / 255
X_valid = X_valid.astype('float32') / 255
X_test = X_test.astype('float32') / 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

base_model = 'inception_resnet_v2'
if 'inception_resnet_v2' in base_model:
    model = load_model('../h5/inceptionResnetV2_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
if 'inception_v3' in base_model:
    model = load_model('../h5/inceptionv3_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
if 'vgg' in base_model :
    model = load_model('../h5/vgg16_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
if 'resnet_50' in base_model:
    model = load_model('../h5/resnet50_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
optimizer = SGD(lr=1e-3)
model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
model.summary()

y_pred = model.predict(X_test)
print("======================y_test = ", y_test)

y_test_list = np.argmax(y_test, axis=1)
y_pred_list = np.argmax(y_pred, axis=1)
# print("============y_test_list=", y_test_list)
# print("==========y_pred_list=",  y_pred_list)

precision = precision_score(y_test_list, y_pred_list, average='micro')
recall = recall_score(y_test_list, y_pred_list, average = 'micro')
f1 = f1_score(y_test_list, y_pred_list, average = 'micro')
acc = accuracy_score(y_test_list, y_pred_list)

print('Precision Score:')
print(precision)

print('Recall Score:')
print(recall)

print('F1 Score: ')
print(f1)

print('Classification report:')
print(classification_report(y_test_list, y_pred_list, digits=4))

print('Accuracy score: ')
print(acc)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i]) # calculate ROC
    roc_auc[i] = auc(fpr[i], tpr[i]) # calculate auc

# micro: Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# macro: Compute macro-average ROC curve and ROC area
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= num_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] =  auc(fpr["macro"], tpr["macro"])

# Plot all ROC curve
lw = 1
plt.figure()
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='Maroon', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]), color='deeppink', linestyle=':', linewidth=4)

# draw each class curve
colors = ['Turquoise','YellowGreen','yellow','Tomato','SpringGreen','SlateBlue','lightpink','Black','thistle','Blue','BlueViolet','lightblue','BurlyWood','SaddleBrown','gold','Chocolate','Coral','red','DarkGray','Crimson','lightcyan','DarkBlue','DarkCyan','DarkGoldenRod','Fuchsia','DarkGreen','DarkKhaki','DarkMagenta','linen','Orange','orangered','DarkRed','DarkSalmon','DarkSeaGreen','DarkSlateBlue','DarkSlateGray']

for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color = color, lw = lw, label = 'ROC curve of {0} (area = {1:0.2f})'.format(class_names[i], roc_auc[i]))
    
plt.plot([0,1], [0,1], 'k--', lw = lw)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

font_style = {'family' : 'Arial',
    'weight' : 'normal',
    'size'   : 11,
    }

plt.xlabel('False Positive Rate', font_style)
plt.ylabel('True Positive Rate', font_style)
plt.title('Receiver operating characteristic', font_style)
plt.legend(loc='lower right', ncol=2)
plt.savefig("ROC.png")
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
