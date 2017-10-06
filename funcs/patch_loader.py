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
directories = ["train","validation"]
initial_path = "/home/eduardo/dataset_test/"

list_of_files = dict()
indexes = dict()
list_of_files_per_split = dict()
list_of_labels_per_split = dict()
total_samples = -1

size = -1
shape = -1
batch_objx = None
batch_objy = None
flag_needs_reseting = False # USED WHEN ITS DIMENSION IS ALTERED

def define(initial_path,classes,batch_size,_shape):
    global list_of_files,indexes,list_of_files_per_split,list_of_labels_per_split
    global total_samples,size,shape,batch_objx,batch_objy,flag_needs_reseting
      
    list_of_files = dict()
    indexes = dict()
    list_of_files_per_split = dict()
    list_of_labels_per_split = dict()
    total_samples = -1

    size = -1
    shape = -1
    batch_objx = None
    batch_objy = None
    flag_needs_reseting = False # USED WHEN ITS DIMENSION IS ALTERED

    _define_initial_path(initial_path)
    _define_classes(classes)
    _define_size_shape(batch_size,_shape)    
    
def _define_size_shape(batch_size,inp_shape):
    global batch_objx, batch_objy, per_class_size, size,shape
    size = batch_size
    shape = inp_shape
    assert size%len(classes)==0
    per_class_size = int(size/len(classes))
    batch_objx = np.zeros((size,*shape))
    batch_objy = np.zeros((size))    
    
    for directory in directories:
        for cl in classes:
            list_of_files[directory,cl] = glob.glob(os.path.join(initial_path,directory,cl,"*"))
            indexes[directory,cl] = 0  
    
    _create_per_split_lists()
    return 0
    
    
def _create_per_split_lists():
    global list_of_files, list_of_files_per_split, list_of_labels_per_split
    for split in directories:
        full_split = []
        full_split_y = []
        counter = 0
        for cl in classes:
            full_split+=(list_of_files[split,cl])
            full_split_y.append(np.ones(len(list_of_files[split,cl]))*counter)
            counter+=1
            
        list_of_files_per_split[split] = full_split
        list_of_labels_per_split[split] = np.concatenate(full_split_y)    
    
    
def _define_classes(inp):
    global classes
    classes = inp

def _define_initial_path(inp):
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
        
        
def reset_indexes():
    for directory in directories:
        for cl in classes:
            indexes[directory,cl] = 0  
            
def _iterative_load_full_split(split,counter):
    global flag_needs_reseting,total_samples,batch_objx,batch_objy
    
    max_size = len(list_of_files_per_split[split])

    if size<(max_size-counter):
        samples_to_process = size
    else:
        samples_to_process = max_size-counter
        flag_needs_reseting = True
        
    for i in range(0,samples_to_process):
        batch_objx[i] = load(list_of_files_per_split[split][counter+i])
        
    batch_objy[0:samples_to_process] = list_of_labels_per_split[split][counter:counter+size]
    total_samples = samples_to_process
    
def iterative_load_full_split(split,counter):
    global thread
    thread = threading.Thread(target=_iterative_load_full_split,args=(split,counter,))
    thread.start()    
    if counter+size<len(list_of_files_per_split[split]):
        return counter+size
    return 0
    
def split_size(split):
    size=0    
    for cl in classes:
        size+=len(list_of_files[split,cl])

    print("sebug:",size)
    return size    
        
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
    global thread,flag_needs_reseting,batch_objx,batch_objy
    thread.join()
    retx,rety = batch_objx.copy(),batch_objy.copy()
    if flag_needs_reseting:
        flag_needs_reseting = False
        retx,rety = batch_objx[0:total_samples].copy(),batch_objy[0:total_samples].copy()            
    else:
        retx,rety = batch_objx.copy(),batch_objy.copy()
    
    return retx,rety
    
def randomize(directory, clas):
    random.shuffle(list_of_files[directory,clas])