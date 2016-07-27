import os
import h5py

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import backend as K

import scipy.misc as img
import numpy as np
import matplotlib.pyplot as plt

import DataModel

# Prepare network
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

weights_path = 'vgg16_weights.h5'

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

img_width = 400
img_height = 400

def my_gramm_matrix(x):
    features = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
    gram = features @ features.T
    return gram

def my_style_loss(style, combination):
    S = my_gramm_matrix(style)
    C = my_gramm_matrix(combination)
    channels = 3
    size = img_width * img_height
    return np.sum(np.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def prepareImage(imagePath):
    im = img.imresize(img.imread(imagePath, mode='RGB'), (224, 224)).astype(np.float32)
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im

originalStyleShape = (512, 7, 7)
def extractImageStyle(imagePath):
    im = prepareImage(imagePath)
    out = model.predict(im)
    style = out[0, :, :, :]
    assert style.shape == originalStyleShape
    return style.flatten()

def diffImagesStyles(style1, style2):
    if style1.shape != (512 * 7 * 7,) : print("Test call", style1.shape); return 0
    if style2.shape != (512 * 7 * 7,) : print("Test call next", style2.shape); return 0
    styleDiff = my_style_loss(style1.reshape(originalStyleShape), style2.reshape(originalStyleShape))
    
    return styleDiff
