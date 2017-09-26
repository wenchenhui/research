# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:48:56 2017


IMAGE PROCESSING



@author: eduardo
"""
import scipy
import numpy as np

def filter_img(inp,sigma=2):
    res = scipy.ndimage.filters.gaussian_filter(inp,sigma=sigma)
    return res

def improved_non_maxima_supression(inp):
    inp2 = scipy.ndimage.filters.maximum_filter(inp,size=5)
    return (inp==inp2)*inp    
   
def detections(inp,max_dets_per_img):
    dets = np.stack(np.nonzero(inp),axis=1)
    dets_str = inp[dets[:,0],dets[:,1]]
    dets_args = np.argsort(dets_str)
    dets = dets[dets_args[::-1],:]
    dets_str = dets_str[dets_args[::-1]]

    final_dets = []    
    for i in range(max_dets_per_img):
        final_dets.append([dets[i,:],dets_str[i]])
    return final_dets

def all_pad(img,size,mode="constant"):
    return np.pad(img,(size,size),mode)

def complete_size(img,size):
    res = np.zeros(size)
    res[0:img.shape[0],0:img.shape[1]] =img
    return res
    
def restore_size(img,original_size):
    res = img[0:original_size[0],0:original_size[1]]
    return res
    
def all_pad_remove(img,size):
    return img[size:-size,size:-size]