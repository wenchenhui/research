# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:49:16 2017

@author: eduardo
"""

# TODO TODAY:
# -> experiment multiple times the alexMet experience
# -> inbreast lesion classification
# -> soft invariance
import sys

sys.path.append("/home/eduardo/tese/code/")
import glob
import numpy as np
import scipy.misc
import cv2
import utils as ut
import tensorflow as tf
import cnn_models as models
import random
import os


path = "/var/tmp/inbreast3211798814/mass_patches/"
test_proportion = 0.3
batch_size = [2,224,224,1]
batchx = np.zeros(batch_size)
batchy = np.zeros(batch_size[0],np.int64)
class_weights = np.zeros(2)


    
def data_augment(batch):
    le = batch.shape[0]
    rots = np.random.randint(4,size = le)
    mirrs = np.random.randint(2,size = le)
    for i in range(le):
        batch[i] = np.rot90(batch[i],k=rots[i])
        if mirrs[i]:
            batch[i] = np.fliplr(batch[i])
    return batch
    
def get_batch():
    i=0
    for cla in ["normal","cancer"]:
        arr = array_pointer["train"][cla]
        size = arr.shape[0]
        batchx[i] = arr[np.random.randint(size)]
        batchy[i] = i        
        i += 1
        
    return batchx,batchy
        
def get_test():
    set1 = array_pointer["test"]["normal"]
    set2 = array_pointer["test"]["cancer"]
    tx = np.concatenate((set1,set2))
    ty = np.concatenate((np.zeros(set1.shape[0]),np.ones(set2.shape[0])))
    class_weights[0] = 1/set1.shape[0]
    class_weights[1] = 1/set2.shape[0]
    return tx,ty
    
    
classes_tmp = glob.glob(path+"*")
sets = ["train", "test"]
classes = ["normal","cancer"]

array_pointer = dict()

for st in sets:
    array_pointer[st] = dict()
    for cla in classes:
        
        image_paths = glob.glob(path+"/"+st+"/"+cla+"/*")
        n_samples = len(image_paths)
        image_array = np.zeros((n_samples,batch_size[1],batch_size[2],batch_size[3]))
        for i in range(n_samples):
            patch = np.load(image_paths[i])
            patch = cv2.resize(patch,tuple(batch_size[1:3]))
            image_array[i,:,:,0] = patch
            
        array_pointer[st][cla] = image_array


tx,ty = get_test()

def test(experiment_name,func_model):  
    epochs = 1000
    iterations_per_epoch = 20
    learning_rate = 1e-3
    batch_no_of_samples = batch_size[0]
    
    tf.reset_default_graph()
    alex = func_model(2)
    
    experiment_name = experiment_name
    results = np.zeros((epochs,2))
    
    
    #print("Starting training: "+experiment_name+"\n epochs: {}; iterations_per_epoch: {}; learning_rate: {}; batch_size: {}".format(
    #                                                    epochs,iterations_per_epoch,learning_rate,batch_size))
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        #bar = ut.progress_bar(epochs)
        for epoch in range(epochs):
            
            print("")
            print("Epoch {}:".format(epoch))
            #bar = ut.progress_bar(iterations_per_epoch)
            #bar.tick(i)
            preds_s = np.zeros((batch_no_of_samples*iterations_per_epoch,2))
            y_s = np.zeros(batch_no_of_samples*iterations_per_epoch)
            if epoch!=0:            
                #if epoch%250==0: learning_rate /= 10
                for j in range(iterations_per_epoch):
        
                    batchx,batchy = get_batch()
                    batchx = batchx #data_augmentation
                    # A Solution of the P versus NP Problem
                    _,preds = alex.train(sess,batchx,batchy,learning_rate)
                    #print(preds)
                    preds_s[j*2:j*2+2] = preds
                    y_s[j*2:j*2+2] = batchy
                    #bar.tick()            
                
                train_acc = ut.acc_multiclass(preds_s,y_s,np.ones(2))#class_weights)
                #metrics= ut.ut_all(preds[:,1],batchy)
                #tr_loss= metrics[0]
                #tr_acc =  metrics[1]
                #tr_auc =  metrics[2]
            else: train_acc = 0.50
            #arr_list = []
            #for i in range(0,tx.shape[0],50):
            #    arr_list.append(alex.test(sess,(tx[i:i+50]-128)/255))
                #print("one")
                
            preds = alex.test(sess,tx)#np.concatenate(arr_list,axis=0)
            #print("two")
            test_acc = ut.acc_multiclass(preds,ty,class_weights)
            print(train_acc,test_acc)
            #te_loss,te_acc,te_auc = 0,0,0        
            #metrics = ut.ut_all(preds[:,1],ty)
            #te_loss += metrics[0]
            #te_acc +=  metrics[1]
            #te_auc +=  metrics[2]
            
            results[epoch] = np.array([train_acc,test_acc])
            #results[i] = np.array([tr_loss,te_loss,tr_acc,te_acc,tr_auc,te_auc])
                
            #print("\tLosses:\t\t{:.2f} -> {:.2f}".format(tr_loss,te_loss))
            #print("\tAccuracies:\t{:.2f} -> {:.2f}".format(tr_acc,te_acc))
            #print("\tROC - AUCs:\t{:.2f} -> {:.2f}".format(tr_auc,te_auc))
        
        path = "/var/tmp/"+experiment_name
        np.save(path,results)
        #print("Results saved in: "+path)
        #print("Finished: {} s".format(timer.tick()))
        
        
for experiment in range(0,15,1):
    path = "testing/"+str(experiment)
    os.mkdir("/var/tmp/"+path)
    for func in [models.alexMet,models.alexMet_first_dense]:
        print("Experiment",experiment,func)
        
        test(path+"/"+func.__name__,func)