# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:51:15 2017

Segmentation
@author: eduardo
"""
import numpy as np
import glob
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import binary_dilation as dilate
from scipy.ndimage.measurements import label as label
import os
import funcs.image_processing as iproc

th = 0.8

def show(img):
    plt.imshow(img)
    plt.show()

def compute_mean(ori_image,mask):
    mean = (ori_image*mask).sum()/mask.sum()
    return mean
    
def grow_mask(ori_image,mask):
    neighboors = dilate(mask)-mask
    seed = ori_image*neighboors
    seed = np.unravel_index(seed.argmax(),seed.shape)
    mask[seed] = 1
    return mask
    
def get_dets(htmap, sigma=0.8, thresh=-1, num_dets=40):
    
    #htmap = iproc.filter_img(htmap,sigma)
    htmap = htmap*(htmap>thresh)
    htmap = iproc.improved_non_maxima_supression(htmap)
    #np.save(results_path+"/heatmaps"+str(model_num)+"/"+os.path.basename(dataset_first.files_names[iD]),htmap)
    dets = iproc.detections(htmap,num_dets)
    return dets

if False:
    thresholds = []
    for image_file in glob.glob("/home/eduardo/Results/Batch_Dropout_exp_False_True/pmaps/*.npy"):
        if image_file.split("/")[-1][0]!="m":
            ori_image = np.load(image_file)
            
            ori_image = ori_image*(ori_image>th)
            plt.imshow(ori_image)
            dets = get_dets(ori_image.copy())
            
            
            name = image_file.split("/")[-1].split(".")[0]
            if os.path.isfile("/home/eduardo/Results/Batch_Dropout_exp_False_True/pmaps/m"+name+"_0.npy"):       
                gt = np.load("/home/eduardo/Results/Batch_Dropout_exp_False_True/pmaps/m"+name+"_0.npy")
                
                flag=False            
                for i in range(len(dets)):
                    center = dets[i][0]
                    plt.scatter(center[1],center[0],c="b")
                    if gt[center[0],center[1]] == True:
                        if flag == True:
                            thresholds[-1] = max(thresholds[-1],dets[i][1])
                        else:
                            thresholds.append(dets[i][1])
                            flag=True
                            
                if not flag:
                    thresholds.append(0)
                
                #plt.show()
        

thresholds = []

import pickle as pkl
if True:
    dataset = pkl.load(open("/home/eduardo/data_inbreast_40_deform_elas/dataset_test","rb"))
    for iD in dataset.splits["test"]:
        t=0
        file = "/home/eduardo/Results/Batch_Dropout_exp_False_True/pmaps/"+str(iD)+".npy"
        hmap = np.load(file)
        hmap[0,0] = 1
        masks = glob.glob("/home/eduardo/Results/Batch_Dropout_exp_False_True/pmaps/m"+str(iD)+"_*")
        ground_truth = []
        for mask in masks:
            ground_truth.append(np.load(mask))
        #plt.imshow(hmap)
        #plt.show()
        for gt in  ground_truth:
            #plt.imshow(gt)
            #plt.show()
            thresholds.append((gt*hmap).max())
            t=max(t,(gt*hmap).max())
        plt.imshow(np.clip(hmap-t,0,1))
        plt.show()
            
            
            
            
            

                