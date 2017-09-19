# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:22:54 2017

@author: eduardo
"""

from matplotlib import pyplot as plt
import dicom as dcm
import pickle as pkl
import read_inbreast as readin
import scipy
import scipy.ndimage
import cnn_models as models
import tensorflow as tf
import numpy as np
import utils as ut
import sys
import cnn_lib
import os
import time


"""
TODO:

        Implement this techniques to try and deal with a smaller dataset.
        Start with easy to test.
        After this start testing cascade and boosting
 
image level:
H -> data augmentation (later)
     
     ( probability averaging (gaussian filtering) sigma=2 is usually better )
     ( model averaging )    -> better generalization probabily
     ( dropout )            -> better generalization probabily
     
E -> maximum detections per image -> test with values in [3,5,7,9]

E -> probability averaging (rotations/ flipping) (test with 8 versions and see if it is better)

FILTER SIMMETRY - EXPERIMENTS
"""



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
        
        
class set_wrapper:
    def __init__(self,inv_res_factor=28):
        folder = str(inv_res_factor)+"_INbreast_patches_preprocessed"
        folder = "76_12_INbreast_patches_preprocessed"
        print("Using custom folder")
        self.train_sets = [train_batcher("/home/eduardo/tese/data/"+folder+"/tr_neg_0.npy"),
                  train_batcher("/home/eduardo/tese/data/"+folder+"/tr_pos_0.npy")]
    
        self.test_sets = [test_batcher("/home/eduardo/tese/data/"+folder+"/va_neg_0.npy"),
                 test_batcher("/home/eduardo/tese/data/"+folder+"/va_pos_0.npy")]

        self.test_sets[0].images = np.concatenate((self.test_sets[0].images,
                        np.load("/home/eduardo/tese/data/"+folder+"/te_neg_0.npy")))
                        
        self.test_sets[1].images = np.concatenate((self.test_sets[1].images,
                        np.load("/home/eduardo/tese/data/"+folder+"/te_pos_0.npy")))
            
        self.test_current_set = 0
                        
    def tr_batch(self):
        batchx = np.concatenate((self.train_sets[0].next_batch(12),self.train_sets[1].next_batch(4)))[:,:,:,np.newaxis]
        batchy = np.zeros((16))
        batchy[12:16]=1
        return batchx,batchy
        
    def tr_epoch_finished(self):
        return self.train_sets[1].finished()    
    
    def te_batch(self):
        batchx,_= self.test_sets[self.test_current_set].next_batch(128)
        batchy = np.zeros(batchx.shape[0])
        batchy[:] = self.test_current_set
        if self.test_sets[self.test_current_set].finished():
            self.test_current_set+=1
        return batchx[:,:,:,np.newaxis],batchy
        
    def te_epoch_finished(self):
        if self.test_current_set==2:
            self.test_current_set=0
            return True
        return False


def format_list(values):
    values2=[]
    for val in values:
        values2.append("{:1.3f}".format(val))
    return values2


def filter_img(inp,sigma=2):
    res = scipy.ndimage.filters.gaussian_filter(inp,sigma=sigma)
    return res
    
def non_maxima_supression(inp):
    inp = np.pad(inp,5,"constant")
    for i in range(5,inp.shape[0]-5):
        for j in range(5,inp.shape[1]-5):
            if np.max(inp[i-2:i+3,j-2:j+3]) != inp[i,j]:
                inp[i,j]=0
    return inp[5:-5,5:-5]
    
def improved_non_maxima_supression(inp):
    inp2 = scipy.ndimage.filters.maximum_filter(inp,size=5)
    return (inp==inp2)*inp
    
def detections(inp,max_dets_per_img):
    dets = np.stack(np.nonzero(inp),axis=1)
    dets_str = inp[dets[:,0],dets[:,1]]
    dets_args = np.argsort(dets_str)
    dets = dets[dets_args[::-1],:]
    return dets

tolerance_bit = 400   
def inside(detection, mass):
    bbox = mass[1]
    if detection[1]>bbox[0] and detection[1]<bbox[2] and detection[0]>bbox[1] and detection[0]<bbox[3]:
        return True
    
    center = mass[0]
    if ((center[0]-detection[1])**2+(center[1]-detection[0])**2)<tolerance_bit:
        return True
    return False
    
    
"""
POS HOLDS INFORMATION ABOUT TRUE POSITIVES AND THEIR SUSPICION LEVEL.
FALSE NEGATIVES ARE ALSO ENCODED IN WITH SUSPICION LEVEL 0.
NEGS HOLD INFORMATION ABOUT FALSE POSITIVES
"""
def get_positives_strengths(pmap, masses,debug=True,sigma=0,max_dets_per_img=5):
    if debug:
        pmap = filter_img((pmap>0.5)*pmap,sigma)
    else:
        pmap = filter_img(pmap)
    detection_mask = improved_non_maxima_supression(pmap)
    #detection_mask=(detection_mask>0.5)*detection_mask
    detects = detections(detection_mask,max_dets_per_img)
    pos = np.zeros(len(masses))
    mass_counter = 0
    
    for mass in masses:
        for i in range(detects.shape[0]):
            if inside(detects[i],mass):
                pos[mass_counter] = max(pos[mass_counter],pmap[detects[i,0],detects[i,1]])

        mass_counter+=1
        
    
    
    negs = []
    
    for i in range(detects.shape[0]):
        flag = True # ASSUME IT IS A FALSE POSITIVE
        for mass in masses: 
            if inside(detects[i],mass):
                flag = False
        if flag:
            negs.append(pmap[detects[i,0],detects[i,1]])
    
    negs = np.array(negs)
    return pos,negs
    
def compute_pmap(img,sess,pred_full,x_full):
    img2 = np.pad(img,50,"reflect")
    input_image = np.zeros((1,450,450))
    input_image[0,0:img2.shape[0],0:img2.shape[1]] = img2
    pmap = sess.run(pred_full,{x_full:input_image[:,:,:,np.newaxis]})[:,:,:,1]
    pmap = cnn_lib.mult_unstrided(pmap,times=3)[0,13:13+img.shape[0],13:13+img.shape[1]]
    return pmap


def basic_placeholders():
    x = tf.placeholder(tf.float32,[None,76,76,1])
    phase_train = tf.placeholder(tf.bool,[])
    keep_prob = tf.placeholder(tf.float32,[])
    y = tf.placeholder(tf.int32,[None]) 

    return x,phase_train,keep_prob,y
    
def get_images():
    sets_path = "/home/eduardo/tese/data/splits_info"
    splits = pkl.load(open(sets_path,"rb"))
    tr,va,te = splits
    
    imgs = []
    masses = []

    max_shape = 0
    for path in va+te:
        img = readin.read_img(path)
        img = readin.preprocessing(img,resize_factor=float(1/12))
        imgs.append(img)
        masses.append(readin.get_masses(os.path.basename(path.split("_")[0]),resize_factor=float(1/12)))
        max_shape = max(max_shape,img.shape[0],img.shape[1])
        
    return imgs,masses
    
def full_image_results():
    pos_l = []
    negs_l = []
    for i in range(len(full_images)):
        
        img = full_images[i]
        
        masses_local = masses[i]
        pmap = compute_pmap(img,sess,pred_full,x_full)
        
        
        pos,negs = get_positives_strengths(pmap, masses_local,True, 4,5)
        
        pos_l.append(pos)
        negs_l.append(negs)
            
    pos_l = np.concatenate(pos_l)
    negs_l = np.concatenate(negs_l)
        
    prauc = ut.prauc(pos_l,negs_l)
    first, second = ut.positives_captured(pos_l,negs_l)
        #print(sigma,"-> ",prauc,first,second)
    return prauc,first,second
    
"""
Experiment  -> exponential weights  [x] -> (small improvement)
            -> dropout              [x] -> (small improvement)

"""

learning_rate = 1e-4
n_epochs = 200
all_results = []

keep_prob_value=0.5
    
# MODEL STUFF _______________________________________________________

tf.reset_default_graph()

x,phase_train,keep_prob,y = basic_placeholders()
   
model = models.detector76(x,phase_train,keep_prob)
test_model,maintain_averages_op = models.detector76_ema_weights(x ,model)

pred = tf.nn.softmax(model.out)
pred2 = tf.nn.softmax(test_model.out)    

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=model.out))
loss = loss+models.l2_loss(model,1e-4)
training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.control_dependencies([training_op]):
    train_step = tf.group(maintain_averages_op)

x_full = tf.placeholder(tf.float32,[1,450,450,1])
model_full = models.get_full_image_detector76(test_model,x_full)
pred_full = tf.nn.softmax(model_full.out)

# DATA STUFF _________________________________________________________
full_images, masses = get_images()
set_wrapp = set_wrapper(24)
results = np.zeros((n_epochs,9))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(n_epochs):
        
        t = time.time()
        training_ys, training_outs = [],[]
        testing_ys, testing_outs, testing_outs2 = [],[],[]
        for iteration in range(100):
            #sys.stdout.write("\r \x1b[K Current progress: "+"{:2.1%}".format(iteration/100))
            #sys.stdout.flush()
        
            bx,by = set_wrapp.tr_batch()
            out,_ = sess.run([pred, train_step], feed_dict={x:bx, y:by, keep_prob:0.5, phase_train:True})
            
            training_ys.append(by)
            training_outs.append(out)
            
        tr_loss,tr_acc,tr_auc = ut.ut_all(np.concatenate(training_outs),np.concatenate(training_ys))
            
        while set_wrapp.te_epoch_finished()==False:
            bx,by = set_wrapp.te_batch()
            out2 = sess.run(pred2,feed_dict={x:bx,phase_train:False,keep_prob:1.0})
            testing_ys.append(by)
            #testing_outs.append(out1)
            testing_outs2.append(out2)
            
        #te_loss,te_acc,te_auc = ut.ut_all(np.concatenate(testing_outs),np.concatenate(testing_ys))
        te_loss2,te_acc2,te_auc2 = ut.ut_all(np.concatenate(testing_outs2),np.concatenate(testing_ys))
        
        prauc,f_mass,s_mass = full_image_results()
        #prauc,f_mass,s_mass = 0,0,0
        results[epoch,:] = np.array([tr_loss,tr_acc,tr_auc,te_loss2,te_acc2,te_auc2,prauc,f_mass,s_mass])
        dt = time.time()-t
        
        print("Epoch("+str(int(dt))+"s): ",epoch, results[epoch,:])
        
        
np.save("results_average_dropout_full_76_(12)",results)


"""
results = [] # IN THE END WE HAVE 6x20 results
for inv_res_factor in [12,16,20,24,28,32]:
    if os.path.isfile("/home/eduardo/tese/logs/prauc"+str(inv_res_factor)):
        results.append(pkl.load(open("/home/eduardo/tese/logs/prauc"+str(inv_res_factor),"rb")))
        continue
    
    inner_results = []
    sets_path = "/home/eduardo/tese/data/splits_info"
    splits = pkl.load(open(sets_path,"rb"))
    tr,va,te = splits

    imgs = []
    masses = []
    readin.resize_factor = 1/inv_res_factor
    max_shape = 0
    for path in va+te:
        img = scipy.misc.imresize(readin.read_img(path),float(1/inv_res_factor))
        imgs.append(img)
        masses.append(readin.get_masses(os.path.basename(path.split("_")[0])))
        max_shape = max(max_shape,img.shape[0],img.shape[1])
    
    print(max_shape)
    
    
    """
    #TRAINING STUFF
"""
    tf.reset_default_graph()
    
    x = tf.placeholder(tf.float32,[None,36,36,1])
    phase_train = tf.placeholder(tf.bool,[])
    keep_prob = tf.placeholder(tf.float32,[])
    y = tf.placeholder(tf.int32,[None])    
    x_full = tf.placeholder(tf.float32,[1,450,450,1])
    lr = tf.placeholder(tf.float32,[]) 
    
    model = models.detector36(x,phase_train,keep_prob)
    model_full = models.get_full_image_detector(model,x_full)
    pred_full = tf.nn.softmax(model_full.out)
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=model.out))
    loss = loss+models.l2_loss(model,1e-4)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    pred = tf.nn.softmax(model.out)
    
    set_wrapp = set_wrapper(inv_res_factor)
    bys,outs = [],[]
    
    learning_rate = 1e-4
    
    #with tf.Session() as sess:
    sess = tf.Session()
    if True:
        sess.run(tf.global_variables_initializer())
        for ite in range(25000):
            
            #if (ite+1)%10000==0:
            #    learning_rate /= 2

            sys.stdout.write("\r \x1b[K Iteration: "+str(ite))
            sys.stdout.flush()
            
            bx,by = set_wrapp.tr_batch()
            local_loss=0
            out,local_loss,_ = sess.run([pred,loss,train_step],feed_dict={x:bx,y:by,lr:learning_rate,keep_prob=0.5,phase_train:True})
            
            #print("\n")
            #print(local_loss)
            #print("\n")            
            
            bys.append(by)
            outs.append(out)
            
            if (ite+1)%500==0:
            #if set_wrapp.tr_epoch_finished():
                tr_loss,tr_acc,tr_auc = ut.ut_all(np.concatenate(outs),np.concatenate(bys)) #Only in the end
                bys,outs = [],[]
                              
                while set_wrapp.te_epoch_finished()==False:
                    bx,by = set_wrapp.te_batch()        
                    out = sess.run(pred,feed_dict={x:bx,phase_train:False})
                    bys.append(by)
                    outs.append(out)
                te_loss,te_acc,te_auc = ut.ut_all(np.concatenate(outs),np.concatenate(bys))
                #print(format_list([tr_loss,tr_acc,tr_auc,te_loss,te_acc,te_auc]))
                bys,outs = [],[]
                
                
                for max_dets_per_img in [3,5,7,9]:
                    pos_l = []
                    negs_l = []
                    for i in range(len(imgs)):
                        
                        img = imgs[i]
                        
                        masses_local = masses[i]
                        pmap = compute_pmap(img,sess,pred_full,x_full)
                        pos,negs = get_positives_strengths(pmap, masses_local,True, 2,max_dets_per_img)
                        
                        pos_l.append(pos)
                        negs_l.append(negs)
                    pos_l = np.concatenate(pos_l)
                    negs_l = np.concatenate(negs_l)
                    #print(pos_l)
                    prauc = ut.prauc(pos_l,negs_l)
                    first, second = ut.positives_captured(pos_l,negs_l)
                    print("    ---     ",prauc)
                    inner_results = inner_results + [tr_loss,te_loss,tr_auc,te_auc,prauc,first,second]
                    print(format_list([tr_loss,te_loss,tr_auc,te_auc,prauc,first,second]))
                #plt.imshow(pmap)
                
    pkl.dump(inner_results,open("/home/eduardo/tese/logs/prauc"+str(inv_res_factor),"wb"))
    results.append(inner_results)
pkl.dump(results,open("/home/eduardo/tese/logs/praucs","wb"))


a = np.array(results)
b = np.reshape(a,(6,-1,7))

pr_curves = list()
for i in range(6):
    pr_curves.append(b[i,:,4])
    
maximums = [max(x) for x in pr_curves]
for i in range(6):
    plt.plot(pr_curves[i])
plt.show()
print(maximums)
    
    
    
    
    
""" 
    
    

"""
img = prepare_img(img)
pmap = run_img(img)
pmap = trim_pmap(pmap)
"""

