#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import Sequential, Model,load_model
import math, os, json, sys, re
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
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


from keras.applications import VGG16
import numpy as np
from scipy import ndimage

categories=['carry','clapHands','pickUp','pull','push','sitDown','standUp','throw','walk','waveHands']

def dt(folder1,cat):
    
    dataset=[]
    
    j = 0
    y=[]
    for category in categories:
        for file in os.listdir(folder1+'/'+str(category)):
            #print(file)
            img = load_img(folder1+'/'+str(category)+'/'+file)
            img.thumbnail((224, 224))
            img = img.resize((224,224), Image.ANTIALIAS)
            # Convert to Numpy Array
            x = img_to_array(img)  
            x = x.reshape((224,224,3))
            #print(x.shape)
            #im = Image.open(folder+'/'+str(category)+'/'+file)
            #im=cv2.imread(file)
            #cv2.resize(im, (224,224), interpolation=cv2.INTER_CUBIC)
            #im.resize((224, 224))
            #print('hello1')
            #im.load()
            #print('hello2')
            #data = np.asarray(im,dtype="float32")
            #print(data.shape)
            #print("hello1")
            #data= np.array(im)
            #print(data.shape)
            #x = data.reshape((3,224,224))
            dataset.append(x)
            #print(categories.index(category))
            y.append(categories.index(category))

    dataset=np.array(dataset)
    values = np.array(y)
    #print("All images to array!")
    return dataset,values


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder










model_path1 = '/home/ws2/Documents/finaldata1/seg1/train/head_spinal/models/best_weights2.hdf5'

model1 = load_model(model_path1)
dataset1,values=dt('/home/ws2/Documents/finaldata1/seg1/valid/head_spinal',categories)
model1_pred = model1.predict(dataset1,batch_size=4)
classes = np.argmax(model1_pred,axis=1)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

loss, acc = model1.evaluate(dataset1, onehot_encoded)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
#print(accuracy_score(values,classes))
# print(model_v3.evaluate(X_HPM_test,Y_HPM_test,batch_size=5))

model_path2 = '/home/ws2/Documents/finaldata1/seg1/train/left_hand/models/best_weights2.hdf5'
model2 = load_model(model_path2)
dataset2,values=dt('/home/ws2/Documents/finaldata1/seg1/valid/left_hand',categories)
model2_pred = model2.predict(dataset2,batch_size=4)
classes = np.argmax(model2_pred,axis=1)
loss, acc = model2.evaluate(dataset2, onehot_encoded, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
#print(accuracy_score(values,classes))
# print(model_v3.evaluate(X_HPM_test,Y_HPM_test,batch_size=5))

model_path3 = '/home/ws2/Documents/finaldata1/seg1/train/left_leg/models/best_weights2.hdf5'
model3 = load_model(model_path3)
dataset3,values=dt('/home/ws2/Documents/finaldata1/seg1/valid/left_leg',categories)
model3_pred = model3.predict(dataset3,batch_size=4)
classes = np.argmax(model3_pred,axis=1)
loss, acc = model3.evaluate(dataset3, onehot_encoded, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
#print(accuracy_score(values,classes))
# print(model_v3.evaluate(X_HPM_test,Y_HPM_test,batch_size=5))

model_path4 = '/home/ws2/Documents/finaldata1/seg1/train/right_hand/models/best_weights2.hdf5'
model4 = load_model(model_path4)
dataset4,values=dt('/home/ws2/Documents/finaldata1/seg1/valid/right_hand',categories)
model4_pred = model4.predict(dataset4,batch_size=4)
classes = np.argmax(model4_pred,axis=1)
loss, acc = model4.evaluate(dataset4, onehot_encoded, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
#print(accuracy_score(values,classes))
# print(model_v3.evaluate(X_HPM_test,Y_HPM_test,batch_size=5))
model_path5 = '/home/ws2/Documents/finaldata1/seg1/train/right_leg/models/best_weights2.hdf5'
model5 = load_model(model_path4)
dataset5,values=dt('/home/ws2/Documents/finaldata1/seg1/valid/right_leg',categories)
model5_pred = model5.predict(dataset5,batch_size=4)
classes = np.argmax(model5_pred,axis=1)
loss, acc = model5.evaluate(dataset5, onehot_encoded, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

from sklearn.metrics import accuracy_score

import pandas as pd








# Y_test = label_binarize(Y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])=onehot_encoded

## ROC curve for max late fusion


# In[13]:


print('weighted average')

np.random.seed(123)

df = pd.DataFrame(columns=('w1', 'w2', 'w3','w4','w5', 'accuracy'))
#prediction= (model1_pred+model2_pred+model3_pred+model4_pred)/4.0
#classes_pro = prediction_weight_avg.argmax(axis=-1)
#print(accuracy_score(values,classes_pro))
#print(prediction[0])
i = 0
for w1 in range(1,6):
    for w2 in range(1,6):
        for w3 in range(1,6):
            for w4 in range(1,6):
                for w5 in range(1,6):
                    #w=(w1+w2+w3+w4+w5)/5.0
                    w=(w1+w2+w3+w4+w5)
                    prediction_weight_avg = (w1*model1_pred+w2*model2_pred+w3*model3_pred+w4*model4_pred+w5*model5_pred)/w
#                     print(prediction_weight_avg[0])
#                     print("prediction_weight_avg")
#                     print(prediction_weight_avg)
                    classes_pro = np.argmax(prediction_weight_avg,axis=-1)
                    ac1=accuracy_score(values,classes_pro)
                    if(ac1==1.000000):
                        np.save('/home/ws2/Documents/modelsarray/prediction_weight_avg_seg1.npy', prediction_weight_avg)
                        break
                    #print(ac1)
                    df.loc[i] = [w1, w2, w3,w4,w5, ac1]
                    i += 1
print(ac1)
#df.sort(columns=['accuracy'], ascending=False)

df.sort_values(by='accuracy',ascending=False)


# In[17]:



s1 = np.load('/home/ws2/Documents/modelsarray/prediction_weight_avg_seg1.npy')
print(s1)
s2 = np.load('/home/ws2/Documents/modelsarray/prediction_weight_avg_seg2.npy')
s3 = np.load('/home/ws2/Documents/modelsarray/prediction_weight_avg_seg3.npy')
s4 = np.load('/home/ws2/Documents/modelsarray/prediction_weight_avg_seg4.npy')

for w1 in range(1,5):
    for w2 in range(1,5):
        for w3 in range(1,5):
            for w4 in range(1,5):
                w=(w1+w2+w3+w4)
                prediction_weight_avg = (w1*s1+w2*s2+w3*s3+w4*s4)/w
#                     print(prediction_weight_avg[0])
#                     print("prediction_weight_avg")
#                     print(prediction_weight_avg)
                classes_pro = np.argmax(prediction_weight_avg,axis=-1)
                ac1=accuracy_score(values,classes_pro)
                if(ac1==1.000000):
                    np.save('/home/ws2/Documents/modelsarray/final_prediction_weight_avg_allsegs.npy', ac1)
                    break
                #print(ac1)
                #df.loc[i] = [w1, w2, w3,w4,w5, ac1]
                i += 1
print(ac1)


# In[ ]:


class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.xx=0
    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc,top_3_accuracy,top_5_accuracy = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {},top_3_accuracy: {},top_5_accuracy: {}\n'.format(loss,acc,top_3_accuracy,top_5_accuracy))
        if(top_5_accuracy > self.xx):
            model1_pred = model.predict(dataset1,batch_size=4)
            classes = np.argmax(model1_pred,axis=1)
            np.save('/home/ws2/Documents/predarray/pred_seg1_head_spinal.npy', model1_pred)
            print(classes)
            self.xx= top_5_accuracy        
        

