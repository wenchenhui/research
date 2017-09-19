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

classes = ["positive","negative"]
directories = ["train","validation","test"]
initial_path = "I dont know yet"

list_of_files = dict()
indexes = dict()

for directory in directories:
    for cl in classes:
        list_of_files[directory,classes] = glob.glob(os.path.join(initial_path,directory,classes))
        indexes = 0        

size = -1

def define_size_shape(local_size,shape):
    global batch_obj, per_class_size, size
    assert size%len(classes)==0
    per_class_size = int(size/len(classes))
    batch_obj = np.zeros((size,*shape))


batch_obj = None

def _load_random_batch(size, directory):
    images = []
    for clas in classes:
        images.append(list_of_files[directory,clas],per_class_size)
    
    for i in range(size):
        batch_obj[i] = load(images[i])        
        
def load(file):
    img = np.load(file)
    return img
    
def load_random_batch(directory):
    global thread
    thread = threading.Thread(target=_load_random_batch,args=(directory,))
    
def load_next_batch(size, directory):
    return "not yet implemented"

def get_prepared_batch():
    global thread
    thread.join()    
    return batch_obj
    
def randomize(directory, clas):
    random.shuffle(list_of_files[directory,clas])