# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:26:42 2017

@author: eduardo
"""
import sys
sys.path.append("/home/eduardo/research/research/")
import numpy as np
import pickle as pkl
import os
import collections
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from skimage.measure import regionprops,label
from matplotlib import pyplot as plt
import funcs.utils as ut

import cv2
import glob



patch_size = 76
half_patch = int(patch_size/2)
pad = 40
step = 38
transformations = 40
sigma = 20
alpha = 300
use_elastic_deform = False

base_path = "/home/eduardo/inbreast_full_images_12/"
dst_path = "/home/eduardo/inbreast_train_patches_no_deform/"


Transformation = collections.namedtuple("Transformation","angle mirroring shape dx dy")

def _get_random_transformation_parameters(shape):
    angle = np.random.randint(360)
    mirroring = np.random.randint(2)
    
    random_state = np.random.RandomState()
    dx = -1+2*random_state.rand(*shape)
    dy = -1+2*random_state.rand(*shape)
    dx = gaussian_filter(dx,sigma,mode="constant",cval=0) * alpha
    dy = gaussian_filter(dy,sigma,mode="constant",cval=0) * alpha
    
    return Transformation(angle,  mirroring, shape, dx,dy)

    
def _preprocess(image,trans,border = "reflect"):
    
    angle = trans.angle
    mirroring = trans.mirroring
    shape = trans.shape
    dx = trans.dx
    dy = trans.dy
    
    image = image.astype(float)
    
    if use_elastic_deform:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)),np.reshape(x+dx, (-1, 1))
        
        image = map_coordinates(image, indices, order=1).reshape(shape)
    
    rows,cols = image.shape
    
    if mirroring:
        image = cv2.flip(image,0)
    added = int(0.8*rows)+pad
    
    if border == "reflect":
        image = cv2.copyMakeBorder(image,added,added,added,added,cv2.BORDER_REFLECT_101)
    elif border == "constant":
        image = cv2.copyMakeBorder(image,added,added,added,added,cv2.BORDER_CONSTANT)
    else:
        raise(Exception("I don't know that border type",border))
        
    M = cv2.getRotationMatrix2D((image.shape[0]/2,image.shape[1]/2),angle,1)
    image = cv2.warpAffine(image,M,(image.shape[0],image.shape[1]))
    
    return image
    
    
def generate_grid_points(shape):
    xvalues = np.arange(0,shape[0],step)
    yvalues = np.arange(0,shape[1],step)
    xx, yy = np.meshgrid(xvalues, yvalues)
    return np.stack((xx.reshape(-1),yy.reshape(-1)),axis=1)
    
def generate_mask_points(masks):
    
    if len(masks)>0:
        safe_zone = np.zeros(masks[0].shape)
        points = np.zeros((9*len(masks),2),dtype=int)
        i=0
        for m in masks:
            local = get_grid_mask(m)
            points[i*9:(i+1)*9] = local
            safe_zone+=m
    
        return points,safe_zone
        
def get_grid_mask(mask):
    props = regionprops(label(mask))
    area,curr = 0,0
    for i in range(len(props)):
        if props[i].area>area:
            curr=i
            area = props[i].area
    
    bbox = props[curr].bbox
    mx,my,Mx,My = bbox
    pointxx = np.round(np.linspace(mx+(Mx-mx)/4,mx+3*(Mx-mx)/4,3)).astype(int)
    pointyy = np.round(np.linspace(my+(My-my)/4,my+3*(My-my)/4,3)).astype(int)
    
    xx, yy = np.meshgrid(pointxx,pointyy)
    
    return np.stack((xx.reshape(-1),yy.reshape(-1)),axis=1)
    
def generate_safe_zone(masks,roi):
    safe_zone = roi
    for m in masks:
        safe_zone = np.logical_and(safe_zone,np.logical_not(m))
    return safe_zone
    
    
def take_patches(img, safe_zone, points, number,path):
    for i in range(points.shape[0]):
        if safe_zone[points[i,0],points[i,1]]:
            patch = img[points[i,0]-half_patch:points[i,0]+half_patch,points[i,1]-half_patch:points[i,1]+half_patch]
            np.save(path+"/"+str(number),patch)
            number+=1
    
    return number

def load_masks(file):
    mask_files = glob.glob(file+"_m_*")
    masks = []    
    for m in mask_files:
        masks.append(np.load(m))
    return masks

def prepare_img(img):
    return np.pad(img,(pad,pad),"reflect")
    
def prepare_roi(img):
    return np.pad(img,(pad,pad),"constant")
    
def create_train_val_dataset():
    
    data = pkl.load(open(base_path+"/data","rb"))
    
    patch_number = 0    
    os.mkdir(dst_path)
    for split in ["train","validation"]:
        
        os.mkdir(dst_path+split)
        pos_path = dst_path+split+"/positive"
        neg_path = dst_path+split+"/negative"
        os.mkdir(pos_path)
        os.mkdir(neg_path)
        print("Doing: ", split)
        bar = ut.progress_bar(len(data[split]))
        for file in data[split]:
            bar.tick()
            ori_img = np.load(file+".npy")
            ori_roi = np.load(file+"_roi.npy")
            ori_masks = load_masks(file)
            
            for tr in range(transformations):
                transform = _get_random_transformation_parameters(ori_img.shape)
                img = _preprocess(ori_img.copy(),transform)
                roi = _preprocess(ori_roi.copy(),transform,border="constant")
                masks = []
                for mask in ori_masks:
                    masks.append(_preprocess(mask.copy(),transform))
                    
                safe_zone = generate_safe_zone(masks,roi)
                points = generate_grid_points(img.shape)
                
                patch_number = take_patches(img,safe_zone,points,patch_number,neg_path)
            
                masks = []
                for mask in ori_masks:
                    masks.append(_preprocess(mask.copy(),transform,border="constant"))
                    
                if len(masks)>0:
                    points,safe_zone_pos = generate_mask_points(masks)
                    patch_number = take_patches(img,safe_zone_pos,points,patch_number,pos_path)
                    
                
create_train_val_dataset()
        