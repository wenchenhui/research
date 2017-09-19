# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:16:54 2017

@author: eduardo
"""


import read_inbreast as readin
import os 
from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import pickle

patch_size = 76
safe_padding = 40
inv_resize_factor = 12
resize_factor = float(1/inv_resize_factor)
jittering_vector = [-5,0,5]

half_patch_size = int(patch_size/2)



def separate_three_splits(set_,proportions):
    p_tr,p_va = proportions[0],proportions[1]
    
    number_of_elements = len(set_)
    p_tr = round(p_tr * number_of_elements)
    p_va = round(p_va * number_of_elements)
    
    s1 = []
    s2 = []
    s3 = []
    
    for i in range(p_tr):
        s1.append(set_.pop())
        
    for i in range(p_va):
        s2.append(set_.pop())
    
    s3 = list(set_)
    
    return s1,s2,s3

def take_patches(set_, name, buffer_cluster):

    init_time = time.time()    
    print("Started taking patches from set",name,"(",len(set_),")","\n")
    counter_outer=0
    
    for image in set_:
        pat = os.path.basename(image).split("_")[0]
        img_arr = readin.read_img(image)
        #img_arr = readin.resize_image(img_arr)
        img_arr = readin.preprocessing(img_arr,resize_factor)
        
        masses = readin.get_masses(pat,resize_factor)
        safe_zone = img_arr>0
        safe_zone = np.pad(safe_zone,safe_padding,"constant")
        img_arr = np.pad(img_arr,safe_padding,"symmetric")
        
        if len(masses)>0:
            points = np.zeros((len(masses),2))
            
        counter=0
        for mass in masses:
            safe_zone[mass[0][1]-half_patch_size+safe_padding:mass[0][1]+half_patch_size+safe_padding,
                      mass[0][0]-half_patch_size+safe_padding:mass[0][0]+half_patch_size+safe_padding] = 0
            points[counter,0] = mass[0][1]+safe_padding
            points[counter,1] = mass[0][0]+safe_padding
            counter+=1
            
        buffer_cluster.register(name,"pos")
        for mass in masses:
            for offsetx in jittering_vector:
                for offsety in jittering_vector:
                    positive_patch = img_arr[mass[0][1]-half_patch_size+safe_padding+offsetx:mass[0][1]+half_patch_size+safe_padding+offsetx
                                            ,mass[0][0]-half_patch_size+safe_padding+offsety:mass[0][0]+half_patch_size+safe_padding+offsety]

                    buffer_cluster.registered.update(positive_patch)
    
        buffer_cluster.register(name,"neg")
        for i in range(half_patch_size,img_arr.shape[0]-half_patch_size,half_patch_size):
            for j in range(half_patch_size,img_arr.shape[1]-half_patch_size,half_patch_size):
                negative_patch = img_arr[i-half_patch_size:i+half_patch_size,j-half_patch_size:j+half_patch_size]
                buffer_cluster.registered.update(negative_patch)
                
        counter_outer+=1
        sys.stdout.write("\r \x1b[K Current progress: "+"{:2.1%}".format(counter_outer/len(set_)))
        sys.stdout.flush()
        
    buffer_cluster.register(name,"pos")
    buffer_cluster.registered.save()
    buffer_cluster.register(name,"neg")
    buffer_cluster.registered.save()
        
    print("\nFinished:",time.time()-init_time)
        
        
def run(inv_resize_factor):
    folder = str(patch_size)+"_"+str(inv_resize_factor)+"_INbreast_patches_preprocessed"
    os.makedirs("/home/eduardo/tese/data/"+folder+"/")
    buffer_cluster = Buffer_Cluster(50000,"/home/eduardo/tese/data/"+folder+"/",
                                            ["tr_neg","tr_pos","va_neg","va_pos","te_neg","te_pos"])
    take_patches(tr,"tr",buffer_cluster)
    take_patches(va,"va",buffer_cluster)
    take_patches(te,"te",buffer_cluster)
    
    
class Buffer:
    def __init__(self,size,path,name):
        self.path = path
        self.name = name
        self.patches=np.zeros((size,patch_size,patch_size))
        self.counter=0
        self.total=0
    
    def update(self,ptch):
        self.patches[self.counter]=ptch
        self.counter+=1
        if self.needs_save():
            self.save()
    
    def save(self):
        #a=np.sum(np.sum(self.patches[0:self.counter],axis=1),axis=1)
        #assert not any(a==0)
        np.save(self.path+self.name+"_"+str(self.total),self.patches[0:self.counter])
        self.total+=1
        self.counter=0
    
    def not_empty(self):
        if self.counter==0:
            return False
        return True
    
    def needs_save(self):
        if self.counter==self.patches.shape[0]:
            return True
        else:
            return False

class Buffer_Cluster:
    def __init__(self,sizes,paths,names):
        self.tr_neg = Buffer(sizes,paths,names[0])
        self.tr_pos = Buffer(sizes,paths,names[1])
        self.va_neg = Buffer(sizes,paths,names[2])
        self.va_pos = Buffer(sizes,paths,names[3])
        self.te_neg = Buffer(sizes,paths,names[4])
        self.te_pos = Buffer(sizes,paths,names[5])
        self.registered = None
        
    def register(self, set_,lbl):
        if set_=="tr":
            if lbl=="neg":
                self.registered = self.tr_neg
            elif lbl=="pos":
                self.registered = self.tr_pos
        if set_=="va":
            if lbl=="neg":
                self.registered = self.va_neg
            elif lbl=="pos":
                self.registered = self.va_pos

        if set_=="te":
            if lbl=="neg":
                self.registered = self.te_neg
            elif lbl=="pos":
                self.registered = self.te_pos                
              
              
sets_path = "/home/eduardo/tese/data/splits_info"

if not os.path.isfile(sets_path):
    with_masses,no_masses = readin.separate_sets()
    tr,va,te = separate_three_splits(with_masses,[0.6,0.2])
    atr,ava,ate = separate_three_splits(no_masses,[0.6,0.2])
    tr = tr + atr
    va = va + ava
    te = te + ate
    
    splits = tr,va,te
    pickle.dump(splits,open(sets_path,"wb"))

else:
    print("loaded splits")
    splits = pickle.load(open(sets_path,"rb"))
    tr,va,te = splits

run(inv_resize_factor)
"""
for inv_resize_factor in [12,16,20,24,28,32]:
    readin.resize_factor = float(1/inv_resize_factor)
    print("\n\n -------------------------------- \n Running Inverse Factor:",inv_resize_factor)
    run(inv_resize_factor)
"""



                
                
