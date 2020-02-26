#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

categories=['a1','a2','a3','a4','a5','a6','a7','a8','a9']

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











dataset1,values=dt('/home/ws2/Documents/ttfidflore_headspinal/valid',categories)
print(values.shape)


# In[7]:


from sklearn.metrics import accuracy_score

import pandas as pd

import numpy as np
print('weighted average')

np.random.seed(123)

df = pd.DataFrame(columns=('w1', 'w2', 'w3','w4','w5', 'accuracy'))
s1 = np.load('/home/ws2/Documents/flore_predarray/pred_fidflore_headspinal.npy')
# a=np.array([[1,0,0,0,0,0,0,0,0,0]])
# a1=s1[0]
# ss1=np.concatenate((a,s1))
print(s1.shape)
s2 = np.load('/home/ws2/Documents/flore_predarray/pred_fidflore_lefthand.npy')
# b=np.array([[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0]])
# b1=s2[0:3]
# ss2=np.concatenate((b,s2))
print(s2.shape)
s3 = np.load('/home/ws2/Documents/flore_predarray/pred_fidflore_leftleg.npy')
print(s3.shape)
s4 = np.load('/home/ws2/Documents/flore_predarray/pred_fidflore_righthand.npy')

print(s4.shape)
s5 = np.load('/home/ws2/Documents/flore_predarray/pred_fidflore_rightleg.npy')

print(s5.shape)
i=0
for w1 in range(1,6):
    for w2 in range(1,6):
        for w3 in range(1,6):
            for w4 in range(1,6):
                for w5 in range(1,6):
                    w=(w1+w2+w3+w4+w5)
                    prediction_weight_avg = (w1*s1+w2*s2+w3*s3+w4*s4+w5*s5)/w
                    np.save('/home/ws2/Documents/flore_predarray/pred_fidflore_weight_avg.npy',prediction_weight_avg)
    #                     print(prediction_weight_avg[0])
    #                     print("prediction_weight_avg")
    #                     print(prediction_weight_avg)
                    classes_pro = np.argmax(prediction_weight_avg,axis=-1)
                    ac1=accuracy_score(values,classes_pro)
                    if(ac1==1.000000):
#                         np.save('/home/ws2/Documents/predarray/final_prediction_seg1.npy', ac1)
                        break
                    #print(ac1)
                    #df.loc[i] = [w1, w2, w3,w4,w5, ac1]

                    df.loc[i] = [w1, w2, w3,w4,w5, ac1]
                    i += 1
print(ac1)
df.sort_values(by='accuracy',ascending=False)


# In[8]:


def top_n_acc(prediction_weight_avg, values, n):
    best_n = np.argsort(prediction_weight_avg, axis=1)[:,-n:]
#     ts = np.argmax(truths, axis=1)
    successes = 0
    for i in range(values.shape[0]):
        if values[i] in best_n[i,:]:
            successes += 1
    return float(successes)/values.shape[0]

s=top_n_acc(prediction_weight_avg, values, 3)
print(s)
t=top_n_acc(prediction_weight_avg, values, 5)
print(t)


# In[ ]:




