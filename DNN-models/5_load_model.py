# -*- encoding: utf-8 -*-
'''
@File    :   5_load_model.py
@Time    :   2019/12/13 11:15:38
@Author  :   yetaoyu 
@Version :   1.0
@Contact :   taoyuye@gmail.com
@Desc    :   Reproduce the experimental results based on the h5 file and calculate the F1 value
    Decide what model to use by modifying the variable base_model="?"
'''

# here put the import lib
from __future__ import print_function
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
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
import itertools
import sys
import math 
import random
import string
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

fopen_acc = open("history_acc_stop.txt", 'w')
fopen_loss = open("history_loss_stop.txt", 'w')

def plot_confusion_matrix(cm, classes,normalize=False, title="confusion matrix", cmap=plt.cm.Blues):
    plt.figure()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalization
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.array(range(len(classes)))
    plt.xticks(tick_marks, classes, rotation=45) 
    plt.yticks(tick_marks, classes) 

    plt.gca().set_xticks([x + 0.5 for x in range(len(classes))], minor=True)
    plt.gca().set_yticks([y + 0.5 for y in range(len(classes))], minor=True)

    # No frame
    #plt.gca().spines['top'].set_visible(False)
    #plt.gca().spines['right'].set_visible(False)
    #plt.gca().spines['bottom'].set_visible(False)
    #plt.gca().spines['left'].set_visible(False)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if normalize:
            if c > 0.01:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize = 8, va = 'center', ha = 'center')
        else:
            if c > 0:
                plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize = 8, va = 'center', ha = 'center')

    # No scale
    #plt.gca().xaxis.set_ticks_position('none')
    #plt.gca().yaxis.set_ticks_position('none')

    # Adding intermediate grid lines
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    plt.gca().tick_params(which='major', labelbottom='off', labelleft='off') # Hide major tick labels
    plt.gca().tick_params(which='minor', width=0) # Finally, hide minor tick marks

    plt.ylabel('True sample labels', fontsize = 14)
    plt.xlabel('Predicted cancaer types', fontsize = 14)

    plt.tight_layout()
    
    if normalize:
        plt.savefig('Confusion_matrix_normalized_stop.png', dpi = 600)
    else:
        plt.savefig('Confusion_matrix_not_normalized_stop.png', dpi = 600)
        
    plt.show()

def load_data(input_dir, image_width, image_height): # input_dir = train
    file_paths = [] # save figure path
    data = []
    labels = []
    class_names = []
    label = -1

    for img_fold in scandir(input_dir): # eg: img_fold = ACC
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

base_model = 'inception_resnet_v2'
IMAGE_SIZE = 310
num_classes = 36
input_shape = (IMAGE_SIZE,IMAGE_SIZE,3)

if 'inception_resnet_v2' in base_model:
    model = load_model('../h5/inceptionResnetV2_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
if 'inception_v3' in base_model:
    model = load_model('../h5/inceptionv3_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
if 'vgg' in base_model :
    model = load_model('../h5/vgg16_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
if 'resnet_50' in base_model:
    model = load_model('../h5/resnet50_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
model.summary()

path = os.getcwd() + '/dataset'
train_path = path + '/train'
valid_path = path + '/valid'
test_path = path + '/test'
print("train_path = ", train_path)

X_train, y_train, class_names = load_data(train_path, IMAGE_SIZE, IMAGE_SIZE)
X_valid, y_valid, class_names2 = load_data(valid_path, IMAGE_SIZE, IMAGE_SIZE)
X_test, y_test, class_names3 = load_data(test_path, IMAGE_SIZE, IMAGE_SIZE)
print("class_names = ", class_names)
input_shape = X_train.shape[1:]

# Normalize data.
X_train = X_train.astype('float32') / 255
X_valid = X_valid.astype('float32') / 255
X_test = X_test.astype('float32') / 255

print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'valid samples')
print(X_test.shape[0], 'test samples')
print('x_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Score trained model.
print('---final model---')
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

y_pred = model.predict(X_test)
print("======================y_test = ", y_test)

y_test_list = np.argmax(y_test, axis=1)
y_pred_list = np.argmax(y_pred, axis=1)
# print("============y_test_list=", y_test_list)
# print("==========y_pred_list=",  y_pred_list)

precision = precision_score(y_test_list, y_pred_list, average='weighted')
recall = recall_score(y_test_list, y_pred_list, average = 'weighted')
f1 = f1_score(y_test_list, y_pred_list, average = 'weighted')
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

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_list, y_pred_list)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=class_names, title="non_Normalized confusion matrix")
# Plot normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')