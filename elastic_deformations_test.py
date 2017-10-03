# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 18:39:37 2017

@author: eduardo
"""

import configs_funcs as cfunc
import os
import pickle as pkl
import train_loop
import tensorflow as tf


dataset = "inbreast"
inv_scale = 24
clahe = True


first_dataset_name = dataset
use_elastic_deform= True
no_transformations = 40

pre_data_folder = cfunc.get_path_predata(dataset,inv_scale,clahe)

if not os.path.isdir(pre_data_folder):
    cfunc.create_standart_predata(dataset,inv_scale,clahe,debug=False)
    
    
results = dict()
max_pr,max_roc,max_acc = 0,0,0

for use_elastic_deform in [False,True]:
    
    dataset_path = cfunc.get_path_data1(first_dataset_name, use_elastic_deform, no_transformations)

    if not os.path.isdir(dataset_path):
        dataset = cfunc.create_first_dataset(pre_data_folder, first_dataset_name , use_elastic_deform, no_transformations)
    else:
        dataset = cfunc.load_dataset(first_dataset_name, use_elastic_deform, no_transformations)

    for i in range(5):
        
        tf.reset_default_graph()
        experiment_name = "Elastic_transf_exp_"+str(use_elastic_deform)+"_"+str(i)

    
        batch,dropout = False,True
        metrics = train_loop.train_loop(experiment_name,1,dataset_path,batch_norm=batch,dropout=dropout)
            
        max_pr += metrics[1::,6].max()
        max_roc += metrics[1::,7].max()
        max_acc += metrics[1::,8].max()
                
    results[use_elastic_deform] = max_pr/5,max_roc/5,max_acc/5
        
pkl.dump(results,open("/home/eduardo/Results/Elastic_exp_results","wb"))