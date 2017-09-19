# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:35:33 2017

@author: eduardo
"""

import numpy as np
from matplotlib import pyplot as plt

a = np.zeros((50,6))
number_of_results = 9

for i in range(number_of_results):
    b = np.load("/var/tmp/test_"+str(i)+".npy")    
    a += b/number_of_results

for i in range(6):
    plt.plot(b[:,i])
    plt.show()