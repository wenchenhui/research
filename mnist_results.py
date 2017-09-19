# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:57:36 2017

@author: eduardo
"""
import numpy as np
from matplotlib import pyplot as plt

for model_rotat in [False,True]:

    for data_aug in [False,True]:
 
       for data_exp in [False,True]:

            name = "rotat" if model_rotat == True else "normal"
            aug = "Aug" if data_aug == True else "noAug"
            exp = "Exp" if data_exp == True else "noExp"
            
            a = np.load("/var/tmp/Conv_first_dense/mnist_"+name+"_"+aug+"_"+exp+".npy")
            plt.plot(a)
            plt.title("/var/tmp/Conv_first_dense/mnist_"+name+"_"+aug+"_"+exp+".npy"+" -> "+str(np.max(a[:,2])))
            plt.ylim(0,1)
            plt.show()
            
def load(rotat,data_aug,data_exp):
    name = "rotat" if rotat == True else "normal"
    aug = "Aug" if data_aug == True else "noAug"
    exp = "Exp" if data_exp == True else "noExp"
    
    return np.load("/var/tmp/Conv_first_dense/mnist_"+name+"_"+aug+"_"+exp+".npy")