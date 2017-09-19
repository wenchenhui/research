# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:24:58 2017

@author: eduardo
"""

import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
folder = "/home/eduardo/tese/logs/same padding experiments/"

for experiment in [12,16,20,24,28,32]:
    accuracies = np.zeros(153)    

    accs = []    
    for split in range(5):
        a=pkl.load(open(folder+"experiment_results"+str(experiment)+"_"+str(split),"rb"))
        s = np.array(a)
        s = s.reshape([-1,5])
        accuracies += s[:,4]/5
        accs.append(np.max(s[:,4]))
        #plt.plot(np.arange(153),s[:,3])
        #plt.plot(np.arange(153),s[:,2])
        #plt.show()
                
    
    acc = np.mean(accs)
    std = np.std(accs)*2
    formated = [ '%.3f' % elem for elem in accs ]
    print(experiment,":","{:1.3f}".format(acc),"pm","{:1.3f}".format(std),formated)
    #plt.plot(np.arange(153), accuracies,color="r")
    #plt.show()