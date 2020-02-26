#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from mpl_toolkits import mplot3d
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
bone_list=[[4,3],[3,21],[21,5],[5,6],[6,7],[7,8],[8,22],[8,23],[21,9],[9,10],[10,11],[11,12],[12,24],[12,25],[21,2],[2,1],[1,13],[1,17],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20]]
bone_list=np.array(bone_list)-1
def func(file):
    colors=['b','k','r','g','y','c','m','w']
    fig=plt.figure()
    fig.set_size_inches(18.5,10.5)
    ax=fig.add_subplot(111,projection='3d')
    ax.set(xlim=(-1,2),ylim=(-2,2),zlim=(-1,0.75));
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')
    n=0
    c=0
    g=0
    jj=0
    k=1
    x=[]
    y=[]
    z=[]
    n=1
    i=0
    if(os.path.exists(os.path.join('/home/ws2/Downloads/nturgb+d_skeletons',file))):
        with open(os.path.join('/home/ws2/Downloads/nturgb+d_skeletons',file),'r') as f:
            for line in f:
                
                words=line.split()
#                 print(words)
                if(i==0):
                    fr=words[0]
                    jj=jj+4
                    i=i+1
                    continue
                    
                if(i==jj):
                    x.append(float(words[0]))
                    y.append(float(words[1]))
                    z.append(float(words[2]))
                    jj=jj+1
                    if(k<25):
                        k=k+1
                    else:
                        x=np.array(x)
                        y=np.array(y)
                        z=np.array(z)
                        print(x)
                        l=n%8
                        j=colors[l]
                        for bone in bone_list:
                            ax.plot([x[bone[0]],x[bone[1]]],[z[bone[0]],z[bone[1]]],[y[bone[0]],y[bone[1]]],j)
                        jj=jj+4
                        n=n+1

                        k=1
                        x=[]
                        y=[]
                        z=[]
                        
                    i=i+1
                        
        plt.show()

                    
                    
                    
                    
                    
                    
                    
                    
for file in os.listdir('/home/ws2/Downloads/nturgb+d_skeletons'):
    func(file)
    


# In[ ]:




