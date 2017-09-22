# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 10:33:23 2017

@author: eduardo
"""
import numpy as np
from matplotlib import pyplot as plt
import time
import sys
import sklearn.metrics


"""
_______________________________________________________________________________
___________________________________ZILEAN______________________________________
SUPPORT METHODS FOR HELPING WITH SIMPLE GUIS
"""
class timer():
    def __init__(self):
        self.current = time.time()
    
    def tick(self):
        ct = time.time()        
        dt = ct-self.current
        self.current = ct
        return dt
        
class progress_bar():
    def __init__(self,steps=100):
        self.steps=steps
        self.last_step = 0
    def tick(self,progress=None):
        if progress==None:
            self.last_step = self.last_step+1
            progress = self.last_step
        _update_progress(float(progress)/self.steps)
        

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def _update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
        






"""
_______________________________________________________________________________
___________________________________TEST________________________________________
PERFORMANCE METRICS V2
"""
def precision_recall_curve(single_preds,single_labels,show=True,ret=False):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(single_labels, single_preds,pos_label=1)
    if show:
        plt.step(recall, precision, color='b', alpha=0.2,where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
        plt.show()
    if ret:
        return precision,recall
        
def pr_auc(single_preds,single_labels):
    prec,rec = precision_recall_curve(single_preds,single_labels,False,True)
    return sklearn.metrics.auc(rec,prec)
    
def roc_curve(single_preds,single_labels):
    fpr, tpr, _ = sklearn.metrics.roc_curve(single_labels,single_preds)
    plt.step(fpr, tpr, color='b', alpha=0.2,where='post')
    plt.fill_between(fpr, tpr, step='post', alpha=0.2,color='b')
    
def roc_auc(single_preds,single_labels):
    return sklearn.metrics.roc_auc_score(single_labels,single_preds)
    
def binary_accuracy(single_preds,single_labels):
    return sklearn.metrics.accuracy_score(single_labels,np.round(single_preds))
    
def ut_metrics(singled_preds, single_labels, loss):
    pr = pr_auc(singled_preds,single_labels)
    roc = roc_auc(singled_preds,single_labels)
    acc = binary_accuracy(singled_preds,single_labels)
    loss = np.mean(loss)
    return ("Loss", loss),("PR AUC",pr),("ROC AUC",roc),("ACCURACY",acc)
    
"""
_______________________________________________________________________________
___________________________________TEST________________________________________
OLD PERFORMANCE METRICS
"""    
    
    
def acc_multiclass(preds,labels,weights):
    preds = np.argmax(preds,axis=1)
    correct_ones = (preds==labels)
    weight_vector = weights[labels.astype(int)]
    #print((preds==0).sum(),(preds==1).sum(),(preds==2).sum())
    #print((labels==0).sum(),(labels==1).sum(),(labels==2).sum())
    return np.sum(correct_ones*weight_vector)/np.sum(weight_vector)


def AUC(preds, labels):
    preds = preds.copy()
    labels = labels.copy()
    numP = (labels==1).sum()
    numN = (labels==0).sum()
    index = np.argsort(preds)[::-1]
    preds = preds[index]
    labels = labels[index]    
        
    auc = 0.0
    height = 0.0
    tpr = 1.0/numP
    fpr = 1.0/numN
    heights = np.zeros(numP+numN)
    
    for i in range(preds.shape[0]):
        if labels[i] == 1:
            height = height+tpr
        else:
            auc+= height*fpr
        heights[i]=height
        
    return auc,heights,preds

def accuracy(preds,labels):# NEED TO BALANCE THIS SHIT
    preds = preds.copy()
    labels = labels.copy()
    
    preds = np.round(preds)

    result = (preds == labels).sum()/labels.shape[0]
    return result
    
def loss(preds, labels):# NEED TO BALANCE THIS SHIT
    preds = preds.copy()
    labels = labels.copy()
    assert (preds==0).sum() == 0
    
    result = -np.sum(labels*np.log(preds+1e-10)+(1-labels)*np.log(1-preds+1e-10))/labels.shape[0]
    return result
    
    
def ut_all(preds,labels):
    # from one hot encoding to label
    preds = preds[:,1]
    
    l = loss(preds,labels)    
    acc = accuracy(preds,labels)
    auc = AUC(preds,labels)[0]
    
    return ("loss",l),("accuracy",acc),("auc",auc)
    
def prauc(pos,negs):
    Npos = pos.shape[0]
    dets = np.concatenate((pos,negs))
    labels = np.concatenate((np.ones(pos.shape[0]),np.zeros(negs.shape[0])))
    args = np.argsort(dets)
    dets = dets[args[::-1]]
    labels = labels[args[::-1]]
    labels_aux = (labels) * (dets>=0.5)
    precision_vec = np.cumsum(labels_aux)/np.cumsum(np.ones(labels.shape[0]))
    
    prauc = 0
    delta_recall = 1/Npos
    max_prec = 0.0
    test_y=[]
    for i in range(precision_vec.shape[0]-1,-1,-1):
        if labels[i]==1:            
            max_prec = max(precision_vec[i],max_prec)
            test_y.append(max_prec)
            prauc += max_prec*delta_recall
            
    #plt.plot(np.arange(1,len(test_y)+1)/Npos,np.array(test_y)[::-1])
    #plt.show()
    #plt.plot(np.cumsum(labels_aux/Npos),precision_vec,c="r")
    #plt.title("precision - recall area under curve = "+str(prauc)+" -> "+str(labels.shape[0]))
    #plt.xlim(0,1)
    #plt.ylim(0,1)
    #plt.show()
    return prauc
    
def positives_captured(pos,negs):
    Npos = pos.shape[0]
    dets = np.concatenate((pos,negs))
    labels = np.concatenate((np.ones(pos.shape[0]),np.zeros(negs.shape[0])))
    args = np.argsort(dets)
    dets = dets[args[::-1]]
    labels = labels[args[::-1]]
    
    first_row = np.sum(labels[0:164]*dets[0:164]>=0.5)/Npos
    second_row = np.sum(labels[0:164*2]*dets[0:164*2]>=0.5)/Npos
    
    return first_row,second_row
    
        
"""
_______________________________________________________________________________
___________________________________OTHER________________________________________
RANDOM AUXILIARY METHODS
"""

def concatenate_arrays(array_list):
    return np.concatenate(array_list,axis=0)
