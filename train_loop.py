# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 17:20:10 2017

@author: eduardo
"""

import funcs.patch_loader as pl
import tensorflow as tf
import funcs.utils as ut
import cnns.cnn_models as models
import numpy as np
import os


def print_s(string,file):
    print(string)
    print(string,file=file)
    

def train_loop(experiment_name, model_number, dataset_path):
    """
    DEFINE PARAMETERS
    """
    #experiment_name = "test_experiment"
    #model_number = 1 # DEFINES IF THIS IS THE FIRST OR SECOND MODEL IN THE CASCADE
    model = models.detector36() # MODEL ARCHITECTURE
    
    
    # LEARNING PARAMETERS
    learning_rate = 0.0001
    
    training_iterations = int(1e4)  # TRAINING ITERATIONS
    one_epoch_every = 1e2           # ITERATIONS PER EPOCH 
    number_of_epochs = int(training_iterations/one_epoch_every)+1
    
    """
    PATH:
        RESULTS
    """
    if model_number == 1:
        results_path = "~/Results/"+experiment_name
        os.mkdir(results_path)
    
    
    log_file = open(results_path+"/log.txt","w") # LOG FILE
    metrics_save_path = results_path+"/metrics.npy"
    
    save_weights_path = results_path+"/model"+str(model_number)
    os.mkdir(save_weights_path)
    load_weights_path = None
    if model_number == 2:
        load_weights_path = results_path+"/model1"
    
    """
        DATASET
    """
    #dataset_path = "/home/eduardo/dataset_name"
    if model_number == 2:
        dataset_path += "second"
        
    
    
    evaluation_metric_index = 1 #DEFINES THE POSITION OF THE METRIC USED FOR COMPARISSON
    evaluation_metric_objective = 1 #1 in case of Maximization and -1 in case of Minimization (ONLY POSITIVE VALUES)
    number_of_evaluation_metrics = 4
    metrics_array = np.zeros((number_of_epochs,number_of_evaluation_metrics*2+1))
    max_evaluation_metric = -1e9
    
    """
    CONFIGURE PATCH LOADER
    """
    pl.define(dataset_path, ["positive","negative"], 128, (36,36,1))
    pl.load_random_batch("train")
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    acts_list, loss_list, y_list = [],[],[]
    
    
    
    if load_weights_path:
        model.load(sess,load_weights_path)
    
    epoch_counter = 0
    
    for iteration in range(training_iterations):
        
        # GET LOADED BATCH
        batchx,batchy = pl.get_prepared_batch()    
        # START LOADING NEXT BATCH
        pl.load_random_batch("train")
        # TRAIN THE MODEL ON THE CURRENT BATCH    
        loss,acts = model.train(sess,batchx,batchy,learning_rate)
        
        # APPEND ITERATION RESULTS IN THE LISTS
        acts_list.append(acts)
        loss_list.append(loss)
        y_list.append(batchy)
        
        if (iteration+1)%one_epoch_every == 0:
    
            pl.load_specific_batch("validation",0)
            val_acts_list, val_loss_list, val_y_list = [],[],[]
            
            for val_iter in range(pl.number_of_batches("validation")):
    
                val_batchx,val_batchy = pl.get_prepared_batch()
                pl.load_specific_batch("validation",val_iter+1)
                val_loss,val_acts = model.test(sess,val_batchx,val_batchy)
                
                val_acts_list.append(acts)
                val_loss_list.append(loss)
                val_y_list.append(val_batchy)
        
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