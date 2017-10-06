# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:05:57 2017

@author: eduardo
"""
#import cnns.cnn_models as models
#import tensorflow as tf
import pickle as pkl
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import os
import glob


sigma = 20
alpha = 300

def get_transformation(image):
    random_state = np.random.RandomState()
    dx = -1+2*random_state.rand(*image.shape)
    dy = -1+2*random_state.rand(*image.shape)
    dx = gaussian_filter(dx,sigma,mode="constant",cval=0) * alpha
    dy = gaussian_filter(dy,sigma,mode="constant",cval=0) * alpha
    
    return dx,dy
    
def transform(image,trans):
    dx,dy = trans
    
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    indices = np.reshape(y+dy, (-1, 1)),np.reshape(x+dx, (-1, 1))
    image = map_coordinates(image, indices, order=1).reshape(image.shape)
    
    return image
    
def inv_transform(image,trans):
    dx,dy = trans
    
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    indices = np.reshape(y-dy, (-1, 1)),np.reshape(x-dx, (-1, 1))
    image = map_coordinates(image, indices, order=1).reshape(image.shape)
    
    return image
    
def test_img(sess,image,model):
    result = np.zeros(image.shape)
    for j in range(10):
        curr = image.copy()        
        trans = get_transformation(curr)
        out = model.test(sess,transform(curr,trans))
        out = inv_transform(out,trans)
        result+=out
    return result/10
   
"""
aux = models.detector76(False, "model"+str(1), False, batch_norm=False ,dropout=True)
model = models.detector76(True, "model"+str(1), True, batch_norm=False ,dropout=True)

sess = tf.Session() 

aux.load(sess,"/home/eduardo/Results/test10/model1")
data = pkl.load(open("/home/eduardo/inbreast_full_images_12/data","rb"))
imgs = data["test"]

os.mkdir("/home/eduardo/heatmaps4/")
for img in imgs:
    image=np.load(img+".npy")/0.13
    out = test_img(sess,image,model)
    #out = model.test(sess,image)
    np.save("/home/eduardo/heatmaps4/"+os.path.basename(img),out)


"""
mmm = []
for i in range(1,5,1):
    maximums = []
    for ht in glob.glob("/home/eduardo/heatmaps"+str(i)+"/*"):
        base_name = os.path.basename(ht).split(".")[0]
        masks = glob.glob("/home/eduardo/inbreast_full_images_12/"+base_name+"_m_*")
        pmap = np.load(ht)
        mask = np.zeros(pmap.shape)
        for m in masks:
            local = np.load(m)
            mask+= local
            maximums.append((local*pmap).max()/local.max())
            print((local*pmap).max()/local.max())
            
        if len(masks)>3:
            f,(ax1,ax2) = plt.subplots(1,2)
            ax1.imshow(pmap)
            ax2.imshow(mask)
            plt.axis("off")
            plt.show()
    mmm.append(maximums)

"""
for i in range(5):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    image=np.load(imgs[i]+".npy")/0.13
    out = model.test(sess,image)
    print(out.max())
    ax1.imshow(image)
    ax2.imshow(out)
    plt.show()
"""    

    

    
"""
aux.load(sess,"/home/eduardo/Results/test10/model1")

for i in range(5):
    
    ori_image=np.load(imgs[i]+".npy")/0.13
    result = np.zeros(ori_image.shape)
    
    for j in range(10):
        curr = ori_image.copy()        
        trans = get_transformation(curr)
        out = model.test(sess,transform(curr,trans))
        out = inv_transform(out,trans)
        result+=out
        plt.imshow(result/(j+1))
        plt.show()
    result=result/10
    print(result.max())
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(ori_image)
    ax2.imshow(result)
    plt.show()
""" 