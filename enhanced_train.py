# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:39:56 2017

@author: eduardo
"""

import funcs.patch_loader as pl
import tensorflow as tf
import funcs.utils as ut
import cnns.cnn_models as models
import numpy as np
import os
import funcs.image_processing as iproc
from matplotlib import pyplot as plt
import pickle as pkl
from shutil import copyfile
import sys

learning_rate = 0.0001
training_iterations = int(2e4)
one_epoch_every = 1e2           # ITERATIONS PER EPOCH 
number_of_epochs = int(training_iterations/one_epoch_every)+1

evaluation_metric_index = 1 #DEFINES THE POSITION OF THE METRIC USED FOR COMPARISSON
evaluation_metric_objective = 1 #1 in case of Maximization and -1 in case of Minimization (ONLY POSITIVE VALUES)
number_of_evaluation_metrics = 4
metrics_array = np.zeros((number_of_epochs,number_of_evaluation_metrics*2+1))
    
def print_s(string,file):
    print(string)
    print(string,file=file)
    
def define_data(dataset_path):
    pl.define(dataset_path, ["negative","positive"], 128, (76,76,1))
    pl.load_random_batch("train")
    
def train_loop(experiment_name, dataset_path, model_number, model_type = models.detector76):
    
    
    model = model_type(False, "model"+str(model_number), False, batch_norm=False ,dropout=True) # MODEL ARCHITECTURE
    
    results_path = "/home/eduardo/Results/"+experiment_name
    if model_number == 1:        
        os.mkdir(results_path)
    
    log_file = open(results_path+"/log"+str(model_number)+".txt","w") # LOG FILE
    metrics_save_path = results_path+"/metrics"+str(model_number)+".npy"
    
    save_weights_path = results_path+"/model"+str(model_number)
    os.mkdir(save_weights_path)

    metrics_array = np.zeros((number_of_epochs,number_of_evaluation_metrics*2+1))
    max_evaluation_metric = -1e9
    
    define_data(dataset_path)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    acts_list, loss_list, y_list = [],[],[]

    epoch_counter = 0
    
    for iteration in range(training_iterations):
        
        # TRAIN
        batchx,batchy = pl.get_prepared_batch()    
        pl.load_random_batch("train")
        loss,acts = model.train(sess,batchx/0.13,batchy,learning_rate)
        
        # APPEND ITERATION RESULTS IN THE LISTS
        acts_list.append(acts)
        loss_list.append(loss)
        y_list.append(batchy)
        
        if (iteration)%one_epoch_every == 0:
    
            split_counter = pl.iterative_load_full_split("validation",0)
            val_acts_list, val_loss_list, val_y_list = [],[],[]
            
            while True:
                val_batchx,val_batchy = pl.get_prepared_batch()
                split_counter = pl.iterative_load_full_split("validation",split_counter)

                val_loss,val_acts = model.test(sess,val_batchx/0.13,val_batchy)
                
                val_acts_list.append(val_acts)
                val_loss_list.append(val_loss)
                val_y_list.append(val_batchy)
                
                if split_counter == 0:
                    
                    val_batchx,val_batchy = pl.get_prepared_batch()
                    
                    val_loss,val_acts = model.test(sess,val_batchx/0.13,val_batchy)
                
                    val_acts_list.append(val_acts)
                    val_loss_list.append(val_loss)
                    val_y_list.append(val_batchy)
                    
                    break
                
            pl.load_random_batch("train")
              
            # COMPUTE METRICS FOR THE TRAINING SET
            acts_list = ut.concatenate_arrays(acts_list)[:,1]
            loss_list = np.array(loss_list)
            y_list = ut.concatenate_arrays(y_list)
            metrics = ut.ut_metrics(acts_list, y_list,loss_list)
            acts_list, loss_list, y_list = [],[],[]
            
            # COMPUTE METRICS FOR THE VALIDATION SET
            val_acts_list = ut.concatenate_arrays(val_acts_list)[:,1]
            val_loss_list = np.array(val_loss_list)
            val_y_list = ut.concatenate_arrays(val_y_list)
            val_metrics = ut.ut_metrics(val_acts_list, val_y_list,val_loss_list)
            
            # PRINT RESULTS
            result_string = "Iter: "+str(iteration)
            result_string += " | Train: "
            for metric in metrics:
                result_string += (metric[0] + ": " + str(metric[1]) + ", ")

            # PRINT RESULTS
            result_string += " | Validation: "
            for metric in val_metrics:
                result_string += (metric[0] + ": " + str(metric[1]) + ", ")
    
            print_s(result_string,log_file)
            
            # SAVE MODEL IF IT IS THE BEST
            curr_evaluation_metric = evaluation_metric_objective * val_metrics[evaluation_metric_index][1]
            if curr_evaluation_metric>max_evaluation_metric:
                max_evaluation_metric = curr_evaluation_metric
                model.save(sess,save_weights_path)
            
            metrics_array[epoch_counter,0] = epoch_counter
            i=1        
            for metric in metrics+val_metrics:
                metrics_array[epoch_counter,i] = metric[1]
                i+=1
                
            epoch_counter+=1
    
    
    log_file.close()
    np.save(metrics_save_path,metrics_array[0:epoch_counter,:])
    sess.close()
    
    return metrics_array[0:epoch_counter,:]


#train_loop("no_deform_new_model","/home/eduardo/inbreast_train_patches_no_deform/",1)
train_loop(sys.argv[1],sys.argv[2],1)
