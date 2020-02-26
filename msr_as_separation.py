#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import shutil
#as1=['a02','a03','a05','a06','a10','a13','a18','a20']
as2=['a01','a04','a07','a08','a09','a11','a12','a14']
for file in os.listdir('/home/ws2/Documents/msr_leftleg'):
    title=file[0:3]
    if title in as2:
        #shutil.copy('/home/ws2/Documents/msr_leftleg/'+file,'/home/ws2/Documents/msr_leftleg1/'+file)
        shutil.copy('/home/ws2/Documents/msr_leftleg/'+file,'/home/ws2/Documents/msr_leftleg_as2/'+file)


# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
# subsets=['train','valid']


for waste_type in as2:
    #folder=os.path.join('/home/ws2/Documents','fidmsr_leftleg',waste_type)
    folder=os.path.join('/home/ws2/Documents','fidmsr_leftleg_as2',waste_type)
    if not os.path.exists(folder):
        os.makedirs(folder)


            
            
for waste in as2:
    i=0
    for file in os.listdir('/home/ws2/Documents/msr_leftleg_as2'):
    #for file in os.listdir('/home/ws2/Documents/msr_leftleg1'):
    
        if(waste in file):
            
            title=file[0:3]
            title1=file[:-4]
            f=os.path.join('/home/ws2/Documents/msr_leftleg_as2/',file)
            #f=os.path.join('/home/ws2/Documents/msr_leftleg1/',file)
            i=i+1
            #shutil.copy(f,os.path.join('/home/ws2/Documents/fidmsr_leftleg',title,title1+'_'+str(i)+'.png'))
            shutil.copy(f,os.path.join('/home/ws2/Documents/fidmsr_leftleg_as2',title,title1+'_'+str(i)+'.png'))
            i=i+1
            img = Image.open(f)
            img2 = img.transpose(Image.FLIP_LEFT_RIGHT)
            #img2.save('/home/ws2/Documents/fidmsr_leftleg/'+title+'/'+title1+'_'+str(i)+'.png')
            img2.save('/home/ws2/Documents/fidmsr_leftleg_as2/'+title+'/'+title1+'_'+str(i)+'.png')
            i=i+1
            img3=img.rotate(15)
            #img3.save('/home/ws2/Documents/fidmsr_leftleg/'+title+'/'+title1+'_'+str(i)+'.png')
            img3.save('/home/ws2/Documents/fidmsr_leftleg_as2/'+title+'/'+title1+'_'+str(i)+'.png')


# In[12]:


from pathlib import Path
from glob2 import glob
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os
import shutil
import re
import seaborn as sns
import random
subsets=['train','valid']
def split_indices(folder,seed1,seed2):
    n=len(os.listdir(folder))
    full_set=list(range(1,n+1))
    random.seed(seed1)
    train=random.sample(list(range(1,n+1)),int(.8*n))
    remain=list(set(full_set)-set(train))
    random.seed(seed2)
    
    valid = list(set(remain)-set(train))
    
    return(train,valid)

## gets file names for a particular type of trash, given indices
    ## input: waste category and indices
    ## output: file names 
def get_names(s_f,indices):
    file_names=[]
    for file in os.listdir(s_f):
        for i in indices:
#             print(i)
#             print(file)
            if(file.endswith(str(i)+'.png')):
                print(file)
                file_names.append(file)
        
#     print(file_names)
    return(file_names)    

## moves group of source files to another folder
    ## input: list of source files and destination folder
    ## no output
def move_files(source_files,destination_folder):
    for file in source_files:
        shutil.copy(file,destination_folder)
        
for subset in subsets:
    for waste_type in as2:
        #folder=os.path.join('/home/ws2/Documents','ttfidmsr_leftleg',subset,waste_type)
        folder=os.path.join('/home/ws2/Documents','ttfidmsr_leftleg_as2',subset,waste_type)
        if not os.path.exists(folder):
            os.makedirs(folder )       

            
## move files to destination folders for each waste type
for waste_type in as2:
    #source_folder = os.path.join('/home/ws2/Documents/fidmsr_leftleg',waste_type)
    source_folder = os.path.join('/home/ws2/Documents/fidmsr_leftleg_as2',waste_type)
    train_ind, valid_ind= split_indices(source_folder,1,1)
    
    ## move source files to train
    train_names = get_names(source_folder,train_ind)
    train_source_files = [os.path.join(source_folder,name) for name in train_names]
    train_dest = "/home/ws2/Documents/ttfidmsr_leftleg_as2/train/"+waste_type
    move_files(train_source_files,train_dest)
    
    ## move source files to valid
    valid_names = get_names(source_folder,valid_ind)
    valid_source_files = [os.path.join(source_folder,name) for name in valid_names]
    valid_dest = "/home/ws2/Documents/ttfidmsr_leftleg_as2/valid/"+waste_type
    move_files(valid_source_files,valid_dest)


# In[ ]:




