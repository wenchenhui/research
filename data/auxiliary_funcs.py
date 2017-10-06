#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Auxiliary -> 

"""

import dicom
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import scipy 
import skimage.morphology as mo
import skimage.measure as me

in_size = 12
def read_dicom_file(file, allow_wrong_sizes = False):
    try:
        img = dicom.read_file(file).pixel_array
    except:
        if allow_wrong_sizes:
            aux = dicom.read_file(file)
            aux[65536*32736+16].value=aux[65536*32736+16].value[1::]
            img = aux.pixel_array
        else:
            raise(Exception ("Wrong size in the pixel array"))
    return img

_clahe_op = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
def preprocess(img, clahe=True, resize_factor=1/in_size):
    
    if clahe:
        img = _clahe_op.apply(img)    
        
    M,m = img.max(),img.min()
    img = ((img-m)/(M-m))*255
    img = img.astype(np.uint8)
    
    img = cv2.resize(img,(0,0),fx = resize_factor,fy = resize_factor,interpolation=cv2.INTER_AREA)
    
    img, roi = _get_roi(img)
    img = (img.astype(float)-128)/256
    
    return img,roi
    
def _get_roi(img,disk_size = int(60*4/in_size)):
    
    img = np.pad(img,disk_size+1,"reflect")
    hat = mo.white_tophat(img,selem = mo.disk(disk_size))
    pre = img-hat
    mask = pre>=10
    
    labels = me.label(mask,connectivity=2)
    if labels.max()>1:
        reg = me.regionprops(labels)
        selected = 0
        for i in range(len(reg)):
            if reg[i].area > reg[selected].area:
                selected=i
        mask = (labels==reg[selected].label)
        
    mask = mo.dilation(mask,selem=mo.disk(int(disk_size/2)))
    final = img*mask
    rm = disk_size+1
    
    return final[rm:-rm,rm:-rm],mask[rm:-rm,rm:-rm]
    
def preprocess_binary(mask,img_shape):
    return scipy.misc.imresize(mask,img_shape,interp="nearest")

def show(img):
    plt.imshow(img,cmap = "gray")
    plt.show()
    
def save_image(arr, directory, number, typ = "img", suffix=0):
    file = os.path.join(directory,str(number))
    
    if typ == "img":
        fsuffix = ""
    elif typ == "mask":
        fsuffix = "_m_"+str(suffix)
    elif typ == "roi":
        fsuffix = "_roi"
    else:
        raise(Exception("save_img wrong typ: "+typ))
    file = os.path.join(directory,str(number)+fsuffix)
    np.save(file,arr)
    return file