# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:29:21 2017

@author: eduardo
"""

import glob
import os
import random
import threading
import numpy as np
from scipy.misc import imread

classes = ["positive","negative"]
directories = ["train","validation","test"]
initial_path = "/home/eduardo/dataset_test/"

list_of_files = dict()
indexes = dict()

size = -1
batch_objx = None
batch_objy = None

def define_size_shape(batch_size,shape):
    global batch_objx, batch_objy, per_class_size, size
    size = batch_size
    assert size%len(classes)==0
    per_class_size = int(size/len(classes))
    batch_objx = np.zeros((size,*shape))
    batch_objy = np.zeros((size))    
    
    for directory in directories:
        for cl in classes:
            list_of_files[directory,cl] = glob.glob(os.path.join(initial_path,directory,cl,"*"))
            indexes[directory,cl] = 0  
            
def define_classes(inp):
    global classes
    classes = inp

def define_initial_path(inp):
    global initial_path
    initial_path = inp

def _load_random_batch(directory):
    global size
    images = []
    
    counter = 0
    for clas in classes:
        images += random.sample(list_of_files[directory,clas],per_class_size)
        batch_objy[counter*per_class_size:(counter+1)*per_class_size] = counter
        counter+=1
    
    for i in range(size):
        batch_objx[i] = load(images[i])
        
def _load_specific_batch(directory,batch_number):
    global size
    images = []
    
    counter = 0
    last = 0
    for clas in classes:
        new_images = list_of_files[directory,clas][batch_number*per_class_size:(batch_number+1)*per_class_size]
        images += new_images
        batch_objy[last:last+len(new_images)] = counter
        counter+=1
        last+=len(new_images)
        
    for i in range(last):
        batch_objx[i] = load(images[i])
        
def load(file):
    #return np.ones((36,36,1))
    #return imread(file)[:,:,np.newaxis]
    return np.load(file)[:,:,np.newaxis]
    
def number_of_batches(set_):
    return 2
    
def load_specific_batch(directory, batch_number):
    global thread
    thread = threading.Thread(target=_load_specific_batch,args=(directory,batch_number,))
    thread.start()
    
def load_random_batch(directory):
    global thread
    thread = threading.Thread(target=_load_random_batch,args=(directory,))
    thread.start()
    
def load_next_batch(size, directory):
    return "not yet implemented"

def get_prepared_batch():
    global thread
    thread.join()
    return batch_objx.copy(),batch_objy.copy()
    
def randomize(directory, clas):
    random.shuffle(list_of_files[directory,clas])