#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os
import shutil
v=1
p=1
a=1


if(os.path.exists(os.path.join('/home/ws2/Downloads/Florence_3d_actions','Florence_dataset_WorldCoordinates.txt'))):
        with open(os.path.join('/home/ws2/Downloads/Florence_3d_actions','Florence_dataset_WorldCoordinates.txt'),'r') as f:
            for line in f:
                words=line.split()
                v1=int(words[0])
                p1=int(words[1])
                a1=int(words[2])
                
                if(os.path.exists(os.path.join('/home/ws2/Documents/flore_fusk','a'+str(a1)+'s'+str(v1)+'p'+str(p1)+'.png'))):
                    f=os.path.join('/home/ws2/Documents/flore_fusk','a'+str(a1)+'s'+str(v1)+'p'+str(p1)+'.png')
                    shutil.copy(f,'/home/ws2/Documents/flore_fusk1/'+'a'+str(a)+'s'+str(v)+'p'+str(p)+'.png')
                v=v1
                p=p1
                a=a1


# In[ ]:




