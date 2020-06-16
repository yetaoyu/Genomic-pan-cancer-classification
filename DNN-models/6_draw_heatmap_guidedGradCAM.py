# -*- encoding: utf-8 -*-
'''
@File    :   6_draw_heatmap_guidedGradCAM.py
@Time    :   2019/12/13 11:31:50
@Author  :   yetaoyu 
@Version :   1.0
@Contact :   taoyuye@gmail.com
@Desc    :   Generate Guided Grad-CAM heatmap and save it to the "Guided Grad-CAM" folder 
    Decide what model to use by modifying the variable base_model="?"
    Decide generate the heatmap by modifying the variable path="?", eg. path= current_path + '/dataset/test' or  current_path + '/dataset/test/BRCA'
'''

# here put the import lib
from __future__ import print_function

from keras.applications import VGG16, InceptionV3, InceptionResNetV2, ResNet50
from keras.layers.core import Lambda
from keras.models import Sequential,load_model
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np 
import keras
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

K.clear_session()
base_model = 'inception_resnet_v2'

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        if 'inception_resnet_v2' in base_model:
            new_model = load_model('../h5/inceptionResnetV2_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
        if 'inception_v3' in base_model:
            new_model = load_model('../h5/inceptionv3_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
        if 'vgg' in base_model :
            new_model = load_model('../h5/vgg16_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
        if 'resnet_50' in base_model:
            new_model = load_model('../h5/resnet50_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')

    return new_model

def compile_saliency_function(model, activation_layer):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def grad_cam(model, img, x, category_index, layer_name):
    """
    Args:
       model: model
       img: image without preprocess, origin image
       x: image input
       category_index: category index
       layer_name: last convolution layer name
    """
    # get category loss
    class_output = model.output[:, category_index]

    # layer output
    convolution_output = model.get_layer(layer_name).output

    # get gradients
    grads = K.gradients(class_output, convolution_output)[0]

    gradient_function = K.function([model.input], [convolution_output, grads])
    
    output, grads_val = gradient_function([x])
    output, grads_val = output[0], grads_val[0]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # create heat map
    cam = cv2.resize(cam, (x.shape[1], x.shape[2]), cv2.INTER_LINEAR) 
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # Return to BGR [0...255] from the preprocessed image
    # image_rgb = x[0, :]
    # image_rgb -= np.min(image_rgb)
    # image_rgb = np.minimum(image_rgb, 255)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(img)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

def deprocess_image(x):
    if np.ndim(x) > 3:
        x = np.squeeze(x) 

    # convert to RGB array
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    
    x *= 255
    
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))

    x = np.clip(x, 0, 255).astype('uint8') 

    return x

def read_img(img_path):
    print(img_path)
    img = cv2.imread(img_path)
    pimg = cv2.resize(img, (img_shape[0], img_shape[1]))
    img = pimg
    pimg = cv2.cvtColor(pimg, cv2.COLOR_BGR2RGB)
    pimg = np.expand_dims(pimg, axis=0)
    pimg = pimg.astype('float32') / 255
    return img, pimg 

IMAGE_SIZE = 310
img_shape=(IMAGE_SIZE,IMAGE_SIZE,3)
current_path = os.getcwd()
path = current_path + '/dataset/test/BRCA' # Mutation map folder path eg./dataset/test/BRCA

if 'inception_resnet_v2' in base_model:
    model = load_model('../h5/inceptionResnetV2_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
    last_conv_layer = 'conv_7b' # the last convolutional layer name 
if 'inception_v3' in base_model:
    model = load_model('../h5/inceptionv3_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
    last_conv_layer = 'conv2d_94'
if 'vgg' in base_model :
    model = load_model('../h5/vgg16_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
    last_conv_layer = 'block5_conv3'
if 'resnet_50' in base_model:
    model = load_model('../h5/resnet50_SGD1e-3_noselectDataset_reduce_factor0.5_patience3_stop.h5')
    last_conv_layer = 'res5c_branch2c'
model.summary()

for root, dirs, files in os.walk(path):
    for file in files:
        img_path = root + '/' + file
        print("img_path=", img_path)
        heatmap_guided = current_path + '/Guided-GradCAM/' + root.split('/')[-1] + '/'
        if not os.path.exists(heatmap_guided):
            os.makedirs(heatmap_guided)

        img, pimg= read_img(img_path)
        
        predictions = model.predict(pimg) # [1 36]
        
        predicted_class = np.argmax(predictions[0])

        cam, heatmap = grad_cam(model, img, pimg, predicted_class, last_conv_layer)
 
        # guided grad_cam img
        register_gradient()
        guided_model = modify_backprop(model, 'GuidedBackProp')
        
        saliency_fn = compile_saliency_function(guided_model, last_conv_layer)
        saliency = saliency_fn([pimg, 0])
        
        gradcam = saliency[0] * heatmap[..., np.newaxis]
        cv2.imwrite(heatmap_guided + file, deprocess_image(gradcam))
     
        # heatmap = np.uint8(255 * heatmap)
        # guided_prop = cv2.resize(np.flip(deprocess_image(saliency[0]),-1), (IMAGE_SIZE, IMAGE_SIZE))
        # guided_gradcam = cv2.resize(np.flip(deprocess_image(gradcam),-1), (IMAGE_SIZE,IMAGE_SIZE))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.subplot(2,3,1), plt.imshow(img), plt.title('origin'), plt.xticks([]), plt.yticks([])
        # plt.subplot(2,3,2), plt.imshow(heatmap), plt.title('heatmap'), plt.xticks([]), plt.yticks([])
        # plt.subplot(2,3,3), plt.imshow(cam), plt.title('Grad-CAM'), plt.xticks([]), plt.yticks([])
        # plt.subplot(2,3,5), plt.imshow(guided_prop), plt.title('Guided backprop'), plt.xticks([]), plt.yticks([])
        # plt.subplot(2,3,6), plt.imshow(guided_gradcam), plt.title('Guided Grad-CAM'), plt.xticks([]), plt.yticks([])
        # plt.show()
