# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:30:04 2017

@author: eduardo
"""

import numpy as np
from matplotlib import pyplot as plt

i = 0

s = np.zeros((1000,2))
no_experiments = 9
for i in range(no_experiments):
    folder = "/var/tmp/testing/"+str(i)+"/"
    a = np.load(folder+"alexMet.npy")
    print(a.max(),a[:,1].max())     
    b = np.load(folder+"alexMet_first_dense.npy")       
    print(b.max(),b[:,1].max())    
    #c = np.concatenate((a,b),axis=1)
    s += a-b

s_val = np.convolve(s[:,1],np.ones(1),mode="SAME")/1
plt.plot(s_val/no_experiments)
#plt.plot(c)
plt.show()