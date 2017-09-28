# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:56:57 2017

@author: eduardo
"""

import data.cbis_standart_format as cbis_standart_format
import simple_data_processor
import train_loop
import pickle as pkl
import os

dst_location = "/home/eduardo/pre_dataset_clahe_"+str(False)+"_"+str(24)+"/"
    
src_location = dst_location
dst_location = "/home/eduardo/dataset_clahe_"+str(False)+"/"

set_masses_size = {"train":491,"validation":100,"test":100}
set_neg_size = {"train":0,"validation":0,"test":0}

if not os.path.isfile(dst_location+"/dataset_test"):    
    dataset = simple_data_processor.make_dataset(set_masses_size, set_neg_size, src_location, dst_location)
    
    halfpatch= 18

    no_transformations = 5
    use_rotations = True
    use_mirroring = True
    use_elastic_deform = False
    
    debug = False
    dataset.make_patches_dataset(no_transformations,use_rotations,use_mirroring,use_elastic_deform,debug)
    dataset.save(dst_location)
else:
    print("Loading dataset")
    dataset = pkl.load(open(dst_location+"/dataset_test","rb"))

#train_loop.train_loop("clahe_"+str(False),1,dst_location)    


th = 0
for dets in [40]:
    for sigma in [0.2,0.5,0.8]:
        sufix = "_"+str(dets)+"_"+str(int(sigma*10))
        all_suspicions = train_loop.test_model(1,"clahe_"+str(False),dataset,sigma,dets,th,sufix)    

