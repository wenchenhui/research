# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 09:21:47 2017

@author: eduardo
"""
#import cnn_lib
import tensorflow as tf
import numpy as np
import sys
import pickle
import os
import cnn_models
import utils


class train_batcher():
    def __init__(self,path):
        self.images = np.load(path)
        self.counter = 0
        self.finished_bool = False
        self.data_aug = True

                
    def next_batch(self,num_samples):
        if self.counter+num_samples>self.images.shape[0]:
            self.counter=0
            self.finished_bool=True
            np.random.shuffle(self.images)            
         
        batch = self.images[self.counter:self.counter+num_samples]
        self.counter+=num_samples
        if self.data_aug:
            batch = self.data_augment(batch)
            
        return batch
        
        
    def data_augment(self,batch):
        le = batch.shape[0]
        rots = np.random.randint(4,size = le)
        mirrs = np.random.randint(2,size = le)
        for i in range(le):
            batch[i] = np.rot90(batch[i],k=rots[i])
            if mirrs[i]:
                batch[i] = np.fliplr(batch[i])
        return batch
    
    def finished(self):
        if self.finished_bool:
            self.finished_bool=False
            return True
        return False
        
class test_batcher():
    def __init__(self,path):
        self.images = np.load(path)
        self.reset()
    
    def size(self):
        return self.images.shape[0]
        
    def reset(self):
        self.counter = 0
                
    def next_batch(self,num_samples):
        num_samples = min(self.images.shape[0]-self.counter,num_samples)
        weight = num_samples/self.images.shape[0]
        batch = self.images[self.counter:self.counter+num_samples]
        self.counter+=num_samples
        return batch,weight
        
    def finished(self):
        if self.counter == self.images.shape[0]:
            self.reset()            
            return True
        return False
    
class tensorboard_wrapper():
    def __init__(self,path):
        self.tb_tr_loss = tf.placeholder(tf.float32,[])
        self.tb_tr_acc = tf.placeholder(tf.float32,[])
        self.tb_te_loss = tf.placeholder(tf.float32,[])
        self.tb_te_acc = tf.placeholder(tf.float32,[])
        self.tb_te_auc = tf.placeholder(tf.float32,[])
        
        tf.summary.scalar("train_loss",self.tb_tr_loss)
        tf.summary.scalar("train_acc",self.tb_tr_acc)
        tf.summary.scalar("test_loss",self.tb_te_loss)
        tf.summary.scalar("test_acc",self.tb_te_acc)
        tf.summary.scalar("test_auc",self.tb_te_auc)
        
        self.merged_summary_op = tf.summary.merge_all()
        
        self.writer = tf.summary.FileWriter(
            "/home/eduardo/tese/logs/tensorflow/"+path, graph=None)
            
    def add_graph(self,sess):
        self.writer.add_graph(sess.graph)
        
    def add_summary(self,sess,values,iteration):
        summ = sess.run(self.merged_summary_op,feed_dict = {
                                                self.tb_tr_loss: values[0],
                                                self.tb_tr_acc: values[1],
                                                self.tb_te_loss: values[2],
                                                self.tb_te_acc: values[3],
                                                self.tb_te_auc: values[4]})
        
        
        self.writer.add_summary(summ,iteration)
    
def run(inv_resize_factor,rep):
     
    tf.reset_default_graph()
    
    folder = str(inv_resize_factor)+"_INbreast_patches"

    x = tf.placeholder(tf.float32,[None,36,36,1])
    phase_train = tf.placeholder(tf.bool,[])
    y = tf.placeholder(tf.float32,[None,2])
    
    logits = cnn_models.detector36(x,phase_train).out
    
    pred = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits))
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
    correct = tf.equal(tf.arg_max(logits,1),tf.arg_max(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    results=[]


    train_sets = [train_batcher("/home/eduardo/tese/data/"+folder+"/tr_neg_0.npy"),
                  train_batcher("/home/eduardo/tese/data/"+folder+"/tr_pos_0.npy")]
    
    test_sets = [test_batcher("/home/eduardo/tese/data/"+folder+"/va_neg_0.npy"),
                 test_batcher("/home/eduardo/tese/data/"+folder+"/va_pos_0.npy")]

    test_sets[0].images = np.concatenate((test_sets[0].images,
                        np.load("/home/eduardo/tese/data/"+folder+"/te_neg_0.npy")))
                        
    test_sets[1].images = np.concatenate((test_sets[1].images,
                        np.load("/home/eduardo/tese/data/"+folder+"/te_pos_0.npy")))

    number_of_iter = 10000

    train_loss,train_acc = 0,0
    test_loss , test_acc = 0,0

    board = tensorboard_wrapper("heyou")
    
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
    
        sess.run(tf.global_variables_initializer())  
        #writer = tf.summary.FileWriter("/home/eduardo/tese/logs/tensorflow/ex"+str(inv_resize_factor)+"_"+str(rep), graph=sess.graph)     
        
        saver.restore(sess,"/home/eduardo/tese/models/last_model-1638")       
        
        training_iter = 0
        for ite in range(number_of_iter):
            
            batchx = np.concatenate((train_sets[0].next_batch(16),train_sets[1].next_batch(16)))[:,:,:,np.newaxis]
            batchy = np.zeros((32,2))
            batchy[0:16,0]=1
            batchy[16:32,1]=1
            training_iter+=1
            #test1,test2 = (sess.run( [logits,modelt],feed_dict={x:batchx,y:batchy}))
            l1,_,l2 = sess.run([loss,train_step,accuracy],feed_dict={x:batchx,y:batchy,phase_train:True})
            #print(test2)
            train_loss+=l1
            train_acc+=l2
    
            sys.stdout.write("\r \x1b[K Iteration: "+str(ite))
            sys.stdout.flush()
            
            
            
            if train_sets[1].finished():
                train_loss/=training_iter
                train_acc/=training_iter
                
                count_preds = 0
                preds = np.zeros(test_sets[0].size()+test_sets[1].size())                
                labels = np.ones(test_sets[0].size()+test_sets[1].size())                
                labels[0:test_sets[0].size()] = 0
                
                for set_ in range(2):
                    test_sets[set_].reset()
                    running=True
                
                    while running:
                        batchx,weights = test_sets[set_].next_batch(128)
                        batchy = np.zeros((batchx.shape[0],2))
                        batchy[:,set_] = 1
                        
                        l1,l2,preds_local = sess.run([loss,accuracy,pred],feed_dict={x:batchx[:,:,:,np.newaxis],y:batchy,phase_train:False})
                        preds[count_preds:count_preds+batchx.shape[0]] = preds_local[:,1]
                        count_preds+=batchx.shape[0]                        
                        
                        
                        test_loss+=l1*weights/2
                        test_acc+=l2*weights/2
                        running = not test_sets[set_].finished()
                
                test_auc,_,_ = utils.AUC(preds,labels)
                print("\n","{:1.3f}".format(train_loss),"{:1.3f}".format(train_acc),"{:1.3f}".format(test_loss),
                      "{:1.3f}".format(test_acc),"{:1.3f}".format(test_auc),"\n")
                
                board.add_summary(sess,[train_loss,train_acc,test_loss,test_acc,test_auc],ite)
                
                results = results+[train_loss,train_acc,test_loss,test_acc,test_auc]
                train_loss,train_acc = 0,0
                test_loss,test_acc = 0,0
                training_iter = 0
            
                saver.save(sess,"/home/eduardo/tese/models/last_model",global_step=ite)
            
    pickle.dump(results,open("/home/eduardo/tese/logs/experiment_results"+str(inv_resize_factor)+"_"+str(rep),"wb"))
    
for scale in [12,16,20,24,28,32]:
    for rep in range(1):
        run(scale,rep)
    
    
    
    
