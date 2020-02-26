#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division,print_function
import math, os, json, sys, re
# import cPickle as pickle
from glob import glob


# from fast.ai.imports import *
import PIL
from PIL import Image
import numpy as np
from numpy.random import random, permutation, randn, normal, uniform, choice
from numpy import newaxis
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
# from scipy.ndimage import imread
#from sklearn.metrics import confusion_matrix
# import bcolz
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

#from IPython.lib.display import FileLink

# import theano
# from theano import shared, tensor as T
# from theano.tensor.nnet import conv2d, nnet
# from theano.tensor.signal import pool

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model,load_model



from keras.layers.core import Flatten, Dense, Dropout, Lambda

from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.optimizers import SGD, RMSprop

import keras.callbacks as kcallbacks


from keras.applications import VGG16,ResNet50
import numpy as np
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from keras.layers.merge import add



class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        
        
        
IMAGE_SIZE = [224, 224]  # we will keep the image size as (64,64). You can increase the size for better results. 

# loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
#vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)  # input_shape = (64,64,3) as required by VGG

#base_model = InceptionResNetV2(input_shape = IMAGE_SIZE + [3],include_top=False )

#base_model = ResNet50(input_shape = IMAGE_SIZE + [3],include_top=False )






# def save_array(fname, arr):
#     c=bcolz.carray(arr, rootdir=fname, mode='w')
#     c.flush()


# def load_array(fname):
#     return bcolz.open(fname)[:]



#path = "/home/ws2/Documents/cropdata/"
path ="/home/ws2/Documents/kinectdata/seg1/"
#path ="/home/ws2/Documents/kinectdata/seg2/"
#path ="/home/ws2/Documents/kinectdata/seg3/"
#path ="/home/ws2/Documents/kinectdata/seg4/"
# path = "data/dogscats/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
    
#batch_size=4

# from vgg16 import Vgg16
# vgg = Vgg16()
#model = vgg
#model = base_model

# In[2]:


# # In[3]:


# # In[6]:

#model.summary()


# In[13]:


#model.layers.pop()
#for layer in model.layers: layer.trainable=True

# trn_data can be array of images in integer form 
# from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam,SGD
# print(model.layers[-1].output.shape)
# h = Flatten()(model.layers[-1].output)
# h = BatchNormalization()(h)
# h = Dense(512, activation='relu')(h)
# h = BatchNormalization()(h)
# h = Activation('relu')(h)
#h=Dense(10, activation='softmax')(h)
from keras.models import Model
#model = Model(input=model.input,output=h)





input_tensor = Input((224, 224, 3))
def block(n_output, upscale=False):
    
    def f(x):
        
        
       
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same')(x)
        
        
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same')(x)
        
        
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        if upscale:
            # 1x1 conv2d
            f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
        else:
            # identity
            f = x
        
        # F_l(x) = f(x) + H_l(x):
        return add([f, h])
    
    return f





x=Conv2D(3,kernel_size=(3,3),strides=(1,1))(input_tensor)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x = block(3)(x)
x=Conv2D(8,kernel_size=(3,3),strides=(1,1))(x)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x = block(8)(x)
x=Conv2D(16,kernel_size=(3,3),strides=(1,1))(x)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x = block(16)(x)
x=Conv2D(32,kernel_size=(3,3),strides=(1,1))(x)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x = block(32)(x)
x=Conv2D(64,kernel_size=(3,3),strides=(1,1))(x)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x = block(64)(x)
x = GlobalAveragePooling2D()(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output)

# In[14]:


gen=image.ImageDataGenerator()
folder = path+'valid'
categories=['carry','clapHands','pickUp','pull','push','sitDown','standUp','throw','walk','waveHands']

training_data= []

onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_files = []
y_train = []
i=0


def dt(folder1,cat):
    
    dataset=[]
    j = 0
    y1=[]
    for category in categories:
        for file in os.listdir(folder1+'/'+str(category)):
            #print(file)
            img = load_img(folder1+'/'+str(category)+'/'+file)
            img.thumbnail((224,224))
            img = img.resize((224,224), Image.ANTIALIAS)
            # Convert to Numpy Array
            x = img_to_array(img)  
            x = x.reshape((224,224,3))
            
            dataset.append(x)
            #print(categories.index(category))
            y1.append(categories.index(category))

    dataset=np.array(dataset)
    values = np.array(y1)
    #print("All images to array!")
    return dataset,values

#print(y)
dataset,values=dt('/home/ws2/Documents/kinectdata/seg1/valid',categories)
#print(dataset.shape)
#print(values.shape)
dataset1,values1=dt('/home/ws2/Documents/kinectdata/seg1/train',categories)

# In[ ]:
#y_labels=onehot(y)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
#print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

integer_encoded1 = label_encoder.fit_transform(values1)
#print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded1 = integer_encoded1.reshape(len(integer_encoded1), 1)
onehot_encoded1 = onehot_encoder.fit_transform(integer_encoded1)




# In[ ]:

#nb_epoch=50
#p=TestCallback((dataset, onehot_encoded))
best_weights_filepath = model_path+'best_weights2.hdf5'
#earlyStopping=kcallbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
#saveBestModel = kcallbacks.ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

#callbacks=[saveBestModel,p]
aug = ImageDataGenerator(rotation_range=0, zoom_range=0,
	width_shift_range=0, height_shift_range=0, shear_range=0,
	horizontal_flip=False)
def fit_model(model, dataset1,onehot_encoded1,dataset,onehot_encoded):
    H = model.fit_generator(aug.flow(dataset1,onehot_encoded1, batch_size=64),
                            validation_data=(dataset,onehot_encoded), steps_per_epoch=len(dataset1) //64,
                            epochs=200,callbacks=[TestCallback((dataset, onehot_encoded))])
    
opt = Adam(lr=0.0001,decay=1e-6)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])




fit_model(model, dataset1,onehot_encoded1,dataset,onehot_encoded)
#model.save_weights(model_path+'after_test_weights.hdf5')

