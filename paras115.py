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
from keras.metrics import top_k_categorical_accuracy


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
from keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout,Add,Multiply,LSTM,Reshape,LeakyReLU
from keras.layers.merge import add
from keras.layers.merge import concatenate


class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc,top_5_accuracy = self.model.evaluate(x, y, verbose=0)
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
#path ="/home/ws2/Documents/finaldata1/seg4/train/head_spinal/"
#path ="/home/ws2/Documents/finaldata1/seg4/train/left_hand/"
#path ="/home/ws2/Documents/finaldata1/seg4/train/left_leg/"
#path ="/home/ws2/Documents/finaldata1/seg4/train/right_hand/"
#path ="/home/ws2/Documents/finaldata1/seg4/train/right_leg/"
path ="/home/ws2/Documents/traintest_dataparts/seg1/train/head_spinal/"
#path ="/home/ws2/Documents/cropdata/train"
#path ="/home/ws2/Documents/traintest_dataparts/seg1/train/left_hand/"
#path ="/home/ws2/Documents/traintest_dataparts/seg1/train/left_leg/"
#path ="/home/ws2/Documents/traintest_dataparts/seg1/train/right_hand/"
#path ="/home/ws2/Documents/traintest_dataparts/seg1/train/right_leg/"
#path ="/home/ws2/Documents/kinectdata/seg2/"
#path ="/home/ws2/Documents/kinectdata/seg3/"
#path ="/home/ws2/Documents/kinectdata/seg4/"
# path = "data/dogscats/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
    

from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam,SGD

from keras.models import Model





input_tensor = Input((224, 224, 3))
def AttentionBlock(x,shortcut,i_filters):
    g1 = Conv2D(i_filters,kernel_size = 1)(shortcut) 
    g1 = BatchNormalization()(g1)
    x1 = Conv2D(i_filters,kernel_size = 1)(x) 
    x1 = BatchNormalization()(x1)

    g1_x1 = Add()([g1,x1])
    psi = Activation('relu')(g1_x1)
    psi = Conv2D(1,kernel_size = 1)(psi) 
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    x = Multiply()([x,psi])
    return x


shortcut=MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(input_tensor)
p=AttentionBlock(input_tensor,shortcut,16)
x1=Conv2D(64,kernel_size=(1,1),padding='same',strides=(1,1))(input_tensor)
x1=BatchNormalization()(x1)
#x1=MaxPooling2D(pool_size=(3,3), strides=(1,1))(x1)
x2=Conv2D(32,kernel_size=(1,1),padding='same',strides=(1,1))(input_tensor)
x2=BatchNormalization()(x2)
x2=Activation('relu')(x2)
#x2=LeakyReLU(alpha=0.1)(x2)
x2=Conv2D(64,kernel_size=(3,3),padding='same',strides=(1,1))(x2)
x2=BatchNormalization()(x2)
#x2=MaxPooling2D(pool_size=(3,3), strides=(1,1))(x2)

x3=Conv2D(32,kernel_size=(1,1),padding='same',strides=(1,1))(input_tensor)
x3=BatchNormalization()(x3)
x3=Activation('relu')(x3)
#x3=LeakyReLU(alpha=0.1)(x3)
x3=Conv2D(64,kernel_size=(3,3),padding='same',strides=(1,1))(x3)
x3=BatchNormalization()(x3)
#x3=MaxPooling2D(pool_size=(3,3), strides=(1,1))(x3)

x4=MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(input_tensor)
x4=Conv2D(32,kernel_size=(1,1),padding='same',strides=(1,1))(x4)
x4=BatchNormalization()(x4)
#x4=MaxPooling2D(pool_size=(3,3), strides=(1,1))(x4)


layer_out1 = concatenate([x1, x2, x3,x4], axis=-1)

x5=Conv2D(224,kernel_size=(1,1),padding='same',strides=(1,1))(p)
#x5=MaxPooling2D(pool_size=(3,3), strides=(1,1))(x5)


h = add([layer_out1, x5])
h=Activation('relu')(h)
#h=LeakyReLU(alpha=0.1)(h)
#x6=Conv2D(32,kernel_size=(1,1),strides=(1,1))(h)
#x6=BatchNormalization()(x6)
#x6=MaxPooling2D(pool_size=(3,3), strides=(1,1))(x6)




#x10=Conv2D(32,kernel_size=(3,3),strides=(1,1))(p)
#x10=MaxPooling2D(pool_size=(3,3), strides=(1,1))(x10)
#h = add([x6, x10])
#h=Activation('relu')(h)
h = GlobalAveragePooling2D()(h)
#print(h.output_shape)
h=Reshape((1,-1))(h)
h = BatchNormalization()(h)


h=LSTM(256)(h)
h=Reshape((1,-1))(h)
h=LSTM(128)(h)
#h=Dense(128)(h)
#h=Activation('relu')(h)
#h=LeakyReLU(alpha=0.1)(h)
#h = BatchNormalization()(h)
output = Dense(10, activation='softmax')(h)

model = Model(inputs=input_tensor, outputs=output)
print(model.summary())
# In[14]:


gen=image.ImageDataGenerator()
folder = path+'valid'
categories=['carry','clapHands','pickUp','pull','push','sitDown','standUp','throw','walk','waveHands']

training_data= []

#onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


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
dataset,values=dt('/home/ws2/Documents/traintest_dataparts/seg1/train/head_spinal',categories)
#print(dataset.shape)
#print(values.shape)
dataset1,values1=dt('/home/ws2/Documents/traintest_dataparts/seg1/valid/head_spinal',categories)

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


def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true,y_pred,k=5)


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true,y_pred,k=3)
# In[ ]:

#nb_epoch=50
#p=TestCallback((dataset, onehot_encoded))
best_weights_filepath = model_path+'aril_best_weights11.hdf5'
#earlyStopping=kcallbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
saveBestModel = kcallbacks.ModelCheckpoint(best_weights_filepath, monitor='val_top_5_accuracy', verbose=1, save_best_only=True, mode='auto')
#,fill_mode='reflect'
#callbacks=[saveBestModel,p]
aug = ImageDataGenerator(rotation_range=30, zoom_range=0.2,fill_mode='nearest',
                         width_shift_range=0.1, height_shift_range=0.1, shear_range=0.01,
                         horizontal_flip=True)
def fit_model(model, dataset1,onehot_encoded1,dataset,onehot_encoded):
    history = model.fit_generator(aug.flow(dataset,onehot_encoded, batch_size=64),
                            validation_data=(dataset1,onehot_encoded1), steps_per_epoch=len(dataset) //64,
                            epochs=700,callbacks=[saveBestModel,TestCallback((dataset1, onehot_encoded1))])
    print(history.history.keys())
    # summarize history for accuracy
    acc=(history.history['top_5_accuracy'])
    
    np.save('/home/ws2/Documents/newmodelsarray/acc_seg1_head_spinal.npy', acc)
    acc1=(history.history['acc'])
    np.save('/home/ws2/Documents/newmodelsarray/acc1_seg1_head_spinal.npy', acc1)
    val_acc=(history.history['val_top_5_accuracy'])
    np.save('/home/ws2/Documents/newmodelsarray/val_acc_seg1_head_spinal.npy', val_acc)
    loss=(history.history['loss'])
    np.save('/home/ws2/Documents/newmodelsarray/loss_seg1_head_spinal.npy', loss)
    val_loss=(history.history['val_loss'])
    np.save('/home/ws2/Documents/newmodelsarray/val_loss_seg1_head_spinal.npy', val_loss)

    
opt = Adam(lr=0.001,decay=1e-6)
#model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc',top_5_accuracy])




fit_model(model, dataset1,onehot_encoded1,dataset,onehot_encoded)
#model.save_weights(model_path+'after_test_weights.hdf5')

