# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 23:45:18 2017
draw images
@author: eduardo
"""
import sys
sys.path.append("/home/eduardo/research/research/")
import numpy as np
import collections
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from matplotlib import pyplot as plt




patch_size = 76
half_patch = int(patch_size/2)
pad = 40
step = 38
transformations = 40
sigma = 80
alpha = 5000
use_elastic_deform = True

base_path = "/home/eduardo/inbreast_full_images_12/"
dst_path = "/home/eduardo/inbreast_beta_patches/"


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
    
    shape = trans.shape
    dx = trans.dx
    dy = trans.dy
    
    image = image.astype(float)
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)),np.reshape(x+dx, (-1, 1))
    
    image = map_coordinates(image, indices, order=1).reshape(shape)
    
    return image

image = np.load("/home/eduardo/inbreast_full_images_12/0.npy")
mask = np.load("/home/eduardo/inbreast_full_images_12/0_m_0.npy")
roi = np.load("/home/eduardo/inbreast_full_images_12/0_roi.npy")

trans = _get_random_transformation_parameters(image.shape)
image[::50,:] = 1
image[:,::50] = 1
mask[::50,:] = 255
mask[:,::50] = 255
roi[::50,:] = 1
roi[:,::50] = 1
dx2 = trans.dx/np.sqrt(trans.dx**2+trans.dy**2)
dy2 = trans.dy/np.sqrt(trans.dx**2+trans.dy**2)
"""
plt.figure(figsize=(image.shape[1]/70,image.shape[0]/70))
plt.axis("off")
plt.quiver(dy2,dx2,scale=80,width = 0.00009,headwidth=0.0000000003,headlength=0.00000000000005,headaxislength=0.00000000000005)
plt.savefig("field")
plt.figure()
plt.axis("off")
"""
plt.imshow(image)
plt.savefig("image1")
plt.figure()
plt.axis("off")
plt.imshow(_preprocess(image,trans))
plt.savefig("image2")
"""
    plt.figure()
    plt.axis("off")
    plt.imshow(mask)
    plt.savefig("mask1")
    plt.figure()
    plt.axis("off")
    plt.imshow(_preprocess(mask,trans))
    plt.savefig("mask2")
    plt.figure()
    plt.axis("off")
    plt.imshow(roi)
    plt.savefig("roi1")
    plt.figure()
    plt.axis("off")
    plt.imshow(_preprocess(roi,trans))
    plt.savefig("roi2")
"""