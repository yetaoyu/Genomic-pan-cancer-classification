# -*- encoding: utf-8 -*-
'''
@File    :   t-SNE.py
@Time    :   2020/05/09 17:24:38
@Author  :   yetaoyu 
@Version :   1.0
@Contact :   taoyuye@gmail.com
@Desc    :   Decide what model to use by modifying the variable base_model="?"
'''
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
from matplotlib.pyplot import cm
import matplotlib.mlab as mlab
from matplotlib.ticker import NullFormatter
import matplotlib.patheffects as PathEffects
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from os import scandir
from keras.models import Model, load_model
import cv2
#from utils import *
# from config import get_config

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
matplotlib.rcParams.update({'font.size': 11}) 

def load_data(input_dir, image_width, image_height): # input_dir = train
    file_paths = [] 
    data = []
    labels = []
    class_names = []
    label = -1 

    for img_fold in scandir(input_dir): # eg. img_fold = ACC
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

def main():
    np.random.seed(42)

    # load data
    IMAGE_SIZE = 310
    num_classes = 36
    input_shape = (IMAGE_SIZE,IMAGE_SIZE,3)

    path = os.getcwd() + '/dataset'
    train_path = path + '/train'
    test_path = path + '/test'

    X_train, y_train, class_names = load_data(train_path, IMAGE_SIZE, IMAGE_SIZE)
    X_test, y_test, class_names3 = load_data(test_path, IMAGE_SIZE, IMAGE_SIZE)
    print("class_names = ", class_names)
    input_shape = X_train.shape[1:]

    # Normalize data.
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    num_classes = len(np.unique(y_test)) 

    # restrict to a sample because slow
    #mask = np.arange(config.num_samples)
    #X_test = X_test[mask].squeeze()
    # y_test = y_test[mask]

    print("X_test: {}".format(X_test.shape))
    print("y_test: {}".format(y_test.shape))

    base_model = 'inception_resnet_v2'
    if 'inception_resnet_v2' in base_model:
        model_load = load_model('../h5/inceptionResnetV2_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
    if 'inception_v3' in base_model:
        model_load = load_model('../h5/inceptionv3_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
    if 'vgg' in base_model :
        model_load = load_model('../h5/vgg16_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
    if 'resnet_50' in base_model:
        model_load = load_model('../h5/resnet50_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')

    model = Model(model_load.input, model_load.layers[-2].output)
    model.summary()
    data=model.predict(X_test, verbose=1)
    embeddings = TSNE(n_components=2, init='pca', perplexity=60, random_state=0).fit_transform(data)
    
    # pickle.dump(embeddings, open(config.data_dir + file_name, "wb"))
    # print("Loading embedding...")
    # embeddings = pickle.load(open(config.data_dir + file_name, "rb"))

    print('Plotting...')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #colors = cm.Spectral(np.linspace(0, 1, num_classes))
    # colors = ['Turquoise','YellowGreen','yellow','Tomato','SpringGreen','SlateBlue','lightpink','Black','thistle','Blue','BlueViolet','lightblue','BurlyWood','SaddleBrown','gold','Chocolate','Coral','RoyalBlue','DarkGray','Crimson','lightcyan','DarkBlue','DarkCyan','DarkGoldenRod','Fuchsia','DarkGreen','DarkKhaki','DarkMagenta','linen','Orange','DarkOrchid','DarkRed','DarkSalmon','DarkSeaGreen','DarkSlateBlue','DarkSlateGray']
    colors = ['Turquoise','YellowGreen','yellow','Tomato','SpringGreen','SlateBlue','lightpink','Black','thistle','Blue','BlueViolet','lightblue','BurlyWood','SaddleBrown','gold','Chocolate','Coral','red','DarkGray','Crimson','lightcyan','DarkBlue','DarkCyan','DarkGoldenRod','Fuchsia','DarkGreen','DarkKhaki','DarkMagenta','linen','Orange','orangered','DarkRed','DarkSalmon','DarkSeaGreen','DarkSlateBlue','DarkSlateGray']
    xx = embeddings[:, 0]
    yy = embeddings[:, 1]

    print("xx=",xx)

    # plot the 2D data points
    for i in range(num_classes):
        print("next=",xx[y_test==i])
        ax.scatter(xx[y_test==i], yy[y_test==i], color=colors[i], label=class_names[i], s=20)

    # we add the labels for each digit
    for i in range(num_classes):
        xtext, ytext = np.median(embeddings[y_test == i, :], axis = 0)
        txt = ax.text(xtext, ytext, class_names[i], fontsize = 11)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground='w'), PathEffects.Normal()])

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.xticks([])
    plt.yticks([])
    # plt.legend(loc='lower left', bbox_to_anchor=(1.01, 0), ncol=1, scatterpoints=1, fontsize=11)
    plt.savefig('t-SNE.png', dpi=300)

    plt.show()

if __name__ == '__main__':
    main()
