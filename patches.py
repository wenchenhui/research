# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:49:16 2017

@author: eduardo
"""

# TODO TODAY:
# -> experiment multiple times the alexMet experience
# -> inbreast lesion classification
# -> soft invariance

import glob
import numpy as np
import scipy.misc
import cv2
import utils as ut
import tensorflow as tf
import cnn_models as models
import random
import os

path = "/var/tmp/0.8368688233757676/patches/"
test_proportion = 0.3
batch_size = [64,224,224,1]
batchx = np.zeros(batch_size)
batchy = np.zeros(batch_size[0],np.int64)
class_weights = np.zeros(3)


def make_class_weights():
    
    all_samples = sum(no_samples_per_class.values())
    for c in class_label:
        class_weights[class_label[c]] = all_samples/no_samples_per_class[c]
    return class_weights
        
def make_tx_ty():
    
    tx = np.zeros((0,*batch_size[1:4]))
    ty = np.zeros(0,np.int64)
    for c in val_images.keys():
        tx = np.concatenate((tx,val_images[c]),axis = 0)
        ty = np.concatenate((ty,np.ones(val_images[c].shape[0])*class_label[c]),axis = 0)

    return tx,ty
    
def load_patch(file):
    return cv2.resize(scipy.misc.imread(file),tuple(batch_size[1:3]))
    
def load_batch():
    to_load = dict()
    
    current = 0
    for c in tr_images.keys():
        #to_load[c] = int(batch_size[0]*no_samples_per_class[c]/sum(no_samples_per_class.values()))
        to_load[c] = int(0.333333*batch_size[0])
        batchx[current:current+to_load[c]] = tr_images[c][np.array(random.sample(range(tr_images[c].shape[0]),to_load[c]))]
        batchy[current:current+to_load[c]] = class_label[c]
        current+=to_load[c]

    # DEBUG THIS
    if current<batch_size[0]:
        remaining = batch_size[0]-current
        random_class = random.choice(list(tr_images.keys()))
        samples_idx = np.array(random.sample(range(tr_images[random_class].shape[0]),remaining))
        batchx[current:batch_size[0]] = tr_images[random_class][samples_idx]
        batchy[current:batch_size[0]] = class_label[c]
        
    return batchx,batchy
    

def data_augment(batch):
    le = batch.shape[0]
    rots = np.random.randint(4,size = le)
    mirrs = np.random.randint(2,size = le)
    for i in range(le):
        batch[i] = np.rot90(batch[i],k=rots[i])
        if mirrs[i]:
            batch[i] = np.fliplr(batch[i])
    return batch
    


classes_tmp = glob.glob(path+"*")
classes = []
for c in classes_tmp:
    classes.append(c.split("/")[-1])
del classes_tmp

no_samples_per_class = dict()
tr_images = dict()
val_images = dict()

class_number = 0
class_label = dict()
for c in classes:
    class_label[c] = class_number
    class_number+=1
    images = glob.glob(path+c+"/*")
    no_samples_per_class[c] = len(images)
    val_samples = int(test_proportion*no_samples_per_class[c])
    val_images[c] = np.zeros((val_samples,*batch_size[1:4]))
    tr_images[c] = np.zeros((no_samples_per_class[c]-val_samples,*batch_size[1:4]))
    
    print(c,":")
    bar=ut.progress_bar(steps=no_samples_per_class[c])
    for i in range(len(images)):
        bar.tick()        
        if i<val_samples:
            val_images[c][i] = load_patch(images[i])[:,:,np.newaxis]
        else:
            tr_images[c][i-val_samples] = load_patch(images[i])[:,:,np.newaxis]

tx,ty = make_tx_ty()
class_weights = make_class_weights()

def test(experiment_name,func_model):  
    epochs = 200
    iterations_per_epoch = 100
    learning_rate = 1e-3
    batch_no_of_samples = batch_size[0]
    
    tf.reset_default_graph()
    alex = func_model(3)
    
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
            preds_s = np.zeros((batch_no_of_samples*iterations_per_epoch,3))
            y_s = np.zeros(batch_no_of_samples*iterations_per_epoch)
            if epoch!=0:            
                #if epoch%250==0: learning_rate /= 10
                for j in range(iterations_per_epoch):
        
                    batchx,batchy = load_batch()
                    batchx = data_augment(batchx-128)/255
                    _,preds = alex.train(sess,batchx,batchy,learning_rate)
                    #print(preds)
                    preds_s[j*64:j*64+64] = preds
                    y_s[j*64:j*64+64] = batchy
                    #bar.tick()            
                
                train_acc = ut.acc_multiclass(preds_s,y_s,np.ones(3))#class_weights)
                #metrics= ut.ut_all(preds[:,1],batchy)
                #tr_loss= metrics[0]
                #tr_acc =  metrics[1]
                #tr_auc =  metrics[2]
            else: train_acc = 0.33333
            arr_list = []
            for i in range(0,tx.shape[0],50):
                arr_list.append(alex.test(sess,(tx[i:i+50]-128)/255))
                #print("one")
                
            preds = np.concatenate(arr_list,axis=0)
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
        
        
for experiment in range(2,10,1):
    path = "experiment_corrected_first_dense/"+str(experiment)
    os.mkdir("/var/tmp/"+path)
    for func in [models.alexMet,models.alexMet_first_dense]:
        print("Experiment",experiment,func)
        
        test(path+"/"+func.__name__,func)