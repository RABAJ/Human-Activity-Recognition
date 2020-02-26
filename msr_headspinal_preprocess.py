#!/usr/bin/env python
# coding: utf-8

# In[ ]:





import os
from mpl_toolkits import mplot3d
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
bone_list=[[19,2],[2,3],[3,6]]
#bone_list=np.array(bone_list)-1
from PIL import Image

import os
# bone_list=[[0,1],[1,2],[1,3],[3,4],[4,5],[1,6],[6,7],[7,8],[2,9],[9,10],[10,11],[2,12],[12,13],[13,14]]
#bone_list=[[3,2],[2,1],[1,0],[0,16],[16,17],[17,18],[18,19],[0,12],[12,13],[13,14],[14,15],[2,4],[4,5],[5,6],[6,7],[2,8],[8,9],[9,10],[10,11]]
#bone_list=np.array(bone_list)-1
def croppingimage(fol1):
#     directory = '/home/paras/Downloads/nturgb/'+str(fol1)
    


    i = Image.open(fol1)
    x1=0
    y1=0
    x2=0
    y2=0
    pixels = i.load() # this is not a list, nor is it list()'able
    width, height = i.size
    print(i.size)
    # print(pixels[1,1])
    all_pixels = []
    f=0
    w=-1
    h=-1
    prev =(255,255,255,255)
    for y in range(height):
        for x in range(width):
            cpixel = pixels[x, y]
            if(cpixel!=(255,255,255,255) and f==0):
    #             print(cpixel)
                f=1
    #             x1=x
                y1=y
            all_pixels.append(cpixel)
            if(cpixel==(255,255,255,255) and prev!=(255,255,255,255)):
    #             x2=w
                y2=h
    #             print(y2,x2)
            prev =cpixel
            w=x
            h=y


    f=0       
    for x in range(width):
        for y in range(height):
            cpixel = pixels[x, y]
            if(cpixel!=(255,255,255,255) and f==0):
    #             print(cpixel)
                f=1
                x1=x
    #             y1=y
            all_pixels.append(cpixel)
            if(cpixel==(255,255,255,255) and prev!=(255,255,255,255)):
                x2=w
    #             y2=h
    #             print(y2,x2)
            prev =cpixel
            w=x
            h=y        
    print(x1,y1,x2,y2) 



    # create a cropped image
    cropped = i.crop((x1,y1,x2,y2))
    # print(pixels[1272,315])
    if(x1!=0):
        cropped.save(fol1)
        # show cropped image
#         cropped.show()
        # print(all_pixels)


def func(file):
    colors=['b','k','r','g','y','c','m','w']
    fig=plt.figure()
    fig.set_size_inches(68.5,60.5)
#     fig1=plt.figure()
#     fig1.set_size_inches(28.5,20.5)
    ax=fig.add_subplot(111,projection='3d')
#     ax.set(xlim=(-1,2),ylim=(-2,2),zlim=(-1,0.75));
    ax.set(xlim=(-2,2),ylim=(3,4),zlim=(-1,1));
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
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
    im=0
    z1=0
    w=0
    v=1
    p=1
    a=1
    title=file[:-4]
    co=0
    
    if(os.path.exists(os.path.join('/home/ws2/Downloads/MSRAction3DSkeletonReal3D',file))):
        with open(os.path.join('/home/ws2/Downloads/MSRAction3DSkeletonReal3D',file),'r') as f:
            for line in f:
                words=line.split()
                
                    
                   
                i=i+1   
                x.append(float(words[0]))
                
                y.append(float(words[1]))
                
                z.append(float(words[2]))
            

    #                     print(jj)
                        
                                        
#                     print(x)
#                     print(y)
#                     print(z)
                if(i%20==0):
                    co=co+1
                    x=np.array(x)
                    y=np.array(y)
                    z=np.array(z)
                    l=n%8
                    j=colors[l]
                    for bone in bone_list:
                        ax.plot([x[bone[0]],x[bone[1]]],[z[bone[0]]+z1,z[bone[1]]+z1],[y[bone[0]],y[bone[1]]],j)
#                         print(bone[0],bone[1])
#                         ax.scatter(m[i,0],m[i,1],m[i,2],color='b') 
#                         ax.text(m[i,0],m[i,1],m[i,2],  '%s' % (str(i)), size=20, zorder=1,  
#                         color='k')
#                     for i in range(0,20):
#                         ax.scatter(x[i],z[i],y[i],color='b')
#                         ax.text(x[i],z[i],y[i], '%s' % (str(i)), size=50, zorder=1, color='k')
                    n=n+1
#                     ax.set_xlabel('x')
#                     ax.set_ylabel('y')
#                     ax.set_zlabel('z')
# #                     ax.view_init(azim=0)
#                     plt.show()
#                     return
                    
                    
                    
#                     return
                    z1=z1+0
                    k=1
                    x=[]
                    y=[]
                    z=[]
#                     return
                
                    
#                     return
                    
                    
#     ax.view_init(azim=0)
#     plt.show()
    print(co)
                    
#     plt.show()                
    plt.savefig('/home/ws2/Documents/msr_headspinal/'+title+'.png',dpi=100)
    croppingimage('/home/ws2/Documents/msr_headspinal/'+title+'.png')                    
    
    
    
for file in os.listdir('/home/ws2/Downloads/MSRAction3DSkeletonReal3D'):
    func(file)
    print(file)


# In[ ]:




