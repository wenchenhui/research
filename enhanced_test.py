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
from skimage.measure import regionprops,label
from skimage.morphology import disk, opening
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
    
def detections(ht,th):
    ht_binary = ht>th
    ht_binary = opening(ht_binary,disk(2))
    props = regionprops(label(ht_binary))
    
    return [x.bbox for x in props]
    
def bbox_binary(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return (rmin, cmin, rmax, cmax)
    
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if (xB-xA)<0 or (yB-yA)<0:
         return 0.0
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou
    
    
data = pkl.load(open("/home/eduardo/inbreast_full_images_12/data","rb"))
imgs = data["test"]
path = "/home/eduardo/heat_maps_with_deform_1only/"

for th in [0.35]:
    ious_full = []
    detections_counter = 0
    for img in imgs:
        base_name = os.path.basename(img).split(".")[0]
        ht = np.load(path+base_name+".npy")
        dets = detections(ht,th)
        
        masks = []
        files = glob.glob(img+"_m_*")
        for mask in files:
            masks.append(np.load(mask))
        
        #plt.imshow(ht)
        #plt.show()
        
        ious = np.zeros(len(masks))
        
        for det in dets:
    
            i=0
            for mask in masks:
                iou = bb_intersection_over_union(det, bbox_binary(mask))
                ious[i] = max(iou,ious[i])
                #print(iou)
                i+=1
                #plt.imshow(mask)
                to_show = np.array([[det[1],det[0]],[det[3],det[0]],[det[3],det[2]],[det[1],det[2]],[det[1],det[0]]])
                #plt.plot(to_show[:,0],to_show[:,1])
                #plt.show()
                
        detections_counter+=len(dets)
        
        for i in range(ious.shape[0]):
            ious_full.append(ious[i])
            plt.scatter(masks[i].sum(),ious[i])
    score1 = (np.array(ious_full)>0.2).sum()
    score2 = detections_counter-score1
    
    print(th,score1,score2)
    plt.show()
    
        

"""
data = pkl.load(open("/home/eduardo/inbreast_full_images_12/data","rb"))
imgs = data["test"]
path = "/home/eduardo/heat_maps_no_deform_safezone_only1/"
for img in imgs:
    base_name = os.path.basename(img).split(".")[0]
    ht = np.load(path+base_name+".npy")
    f,(ax1,ax2)=plt.subplots(1,2)
    ax1.imshow(ht>0.5)
    masks = glob.glob(img+"_m_*")
    mask = np.zeros(ht.shape)
    for m in masks:
        local = np.load(m)
        mask+= local
    ax2.imshow(mask)
    if len(masks)>0:
        plt.show()
        print("hey")


aux = models.detector76(False, "model"+str(1), False, batch_norm=False ,dropout=True)
model = models.detector76(True, "model"+str(1), True, batch_norm=False ,dropout=True)

sess = tf.Session() 

aux.load(sess,"/home/eduardo/Results/new_model_better_safezone/model1")
data = pkl.load(open("/home/eduardo/inbreast_full_images_12/data","rb"))
imgs = data["test"]

os.mkdir("/home/eduardo/heat_maps_no_deform_safezone_only1/")
for img in imgs:
    image=np.load(img+".npy")/0.13
    #out = test_img(sess,image,model)
    out = model.test(sess,image)
    np.save("/home/eduardo/heat_maps_no_deform_safezone_only1/"+os.path.basename(img),out)

""" 

"""
maximums=[]
for ht in glob.glob("/home/eduardo/heat_maps_with_deform_1only/*"):
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