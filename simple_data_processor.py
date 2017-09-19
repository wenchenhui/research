# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:47:37 2017

@author: eduardo


    Dataset class

    -> Responsible for reading and manipulating images
    -> Requires data in the standart format
    
    
    
    Standart format:
    
"""


"""
Basic imports
"""
from matplotlib import pyplot as plt
import numpy as np
import glob
import os
import pickle as pkl
import collections
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass
from scipy.misc import imsave
import cv2
import random
import utils as ut

sqrt2 = np.sqrt(2)
trans_const = (sqrt2-1)/2
halfpatch = 18
"""
Paths setup
"""
split_save_path = "/home/eduardo/dataset_infos/"
dataset_path = "/media/edu/TOSHIBA EXT/dataset2/"
base_patches_path = "/home/edu/dataset_test"

class dataset():
    def __init__(self,iD, name, dataset):
        self.iD = iD
        self.name = name
        self.dataset_name = dataset
        self.files_names = dict()
        self.splits = dict()
        self.masks = dict()
        self.rois = dict()
        
    def add_split(self,name):
        self.splits[name] = set()
        
    def add_image_file(self,split_name,file):
        iD = len(self.files_names)
        self.files_names[iD] = file
        self.splits[split_name].add(iD)
        return iD
    
    def add_mask_file(self,iD,file):
        if iD not in self.masks.keys():
            self.masks[iD] = [file]
        else:
            self.masks[iD].append(file)
    
    def add_roi_file(self,iD,file):
        if iD not in self.rois.keys():
            self.rois[iD] = file
        else:
            raise("It is not working correcly before add_roi_file call!!")
    
    def make_patches_dataset(self,transformations):
        
        counter = 0
        for split in self.splits.keys():
            for img_iD in self.splits[split]:
                
                img = np.load(self.files_names[img_iD])
                masks = load_file_array(self.masks[img_iD])
                roi = np.load(self.rois[img_iD])
                full_mask = sum_masks(masks)
                
                for i in range(transformations):
                    trans = get_random_transformation_parameters()    
                    img = preprocess(img,trans)
                    roi = preprocess_mask(roi,trans)
                    full_mask = preprocess_mask(full_mask,trans)
                    patches = take_negative_patches(full_mask,roi,full_mask)
                    counter = save_patches(split,"negative",patches,counter)
                    
                    for mask in masks:
                        mask = preprocess_mask(mask,trans)
                        patches = take_positive_patches(full_mask,mask)
                        counter = save_patches(split,"positive",patches,counter)

    def save(self):
        pkl.dump(self,open("split_"+self.name,"wb"))
        
        
def save_patches(split,folder,list_of_patches,counter):
    directory = base_patches_path+"/"+split+"/"+folder+"/"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    for p in list_of_patches:
        #np.save(directory+str(counter),p)
        imsave(directory+str(counter)+".png",p)
        counter+=1
    return counter
    
patches = np.zeros((10000,halfpatch*2,halfpatch*2))

def load_file_array(list_of_files):
    result = []
    for file in list_of_files:
        result.append(np.load(file))
    return result

def take_negative_patches(img,roi,full_mask):
    x = np.arange(0,img.shape[0],halfpatch*2) 
    y = np.arange(0,img.shape[1],halfpatch*2)
    
    counter = 0
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            if roi[x[i],y[j]] == 1:
                if full_mask[x[i]-halfpatch:x[i]+halfpatch,y[j]-halfpatch:y[j]+halfpatch].sum()==0:
                    patches[counter] = img[x[i]-halfpatch:x[i]+halfpatch,y[j]-halfpatch:y[j]+halfpatch]
                    counter+=1
    return patches[0:counter].copy()

def take_positive_patches(img,mask):
    centroid = center_of_mass(mask)
    centroid = np.round(centroid).astype(int)
    patch = img[centroid[0]-halfpatch:centroid[0]+halfpatch,centroid[1]-halfpatch:centroid[1]+halfpatch]
    return [patch]


def sum_masks(mask_list):
    total = np.zeros(mask_list[0].shape)
    for i in range(0,len(mask_list)):
        total+=mask_list[i]
    total = np.ceil(total)
    total = np.clip(total,0,1)
    return total
    
def get_image_files(patient):
    return glob.glob(os.path.join(dataset_path,patient+"_img*"))

def get_masks(image_file):
   return glob.glob(image_file.replace("img","mask").split(".")[0]+"_*")

def get_roi(image_file):
    return glob.glob(image_file.replace("img","roi").split(".")[0]+"*")[0]

def has_mass(patient):
    images = get_image_files(patient)
    for image in images:
        if len(get_masks(image)) != 0:
            return True
    return False


def get_patients():
    patients = set()
    files = glob.glob(dataset_path+"*")
    for file in files:
        pat = os.path.basename(file).split("_")[0]
        patients.add(pat)
    
    return list(patients)
    
    
def make_dataset():
    
    set_masses_size = {"train":491,"validation":100,"test":100}
    set_neg_size = {"train":0,"validation":0,"test":0}
    """
    FIND A VALID ID
    """
    splits = glob.glob(os.path.join(split_save_path,"split_*"))
    iDs = []    
    for file in splits:
        split = pkl.load(open(file,"rb"))
        iDs.append(split.iD)
      
    iD=1000
    while (iD in iDs or iD == 1000):
        iD = np.random.randint(1000)
    
    data = dataset(iD,"test","CBIS-DDSM")
    
    """
    FIND ALL PATIENTS
    """
    patients = get_patients()
    
    with_masses, no_masses = [],[]
    for patient in patients:
        if has_mass(patient):
            with_masses.append(patient)
        else:
            no_masses.append(patient)
    
    random.shuffle(with_masses)
    random.shuffle(no_masses)
    
    print("Patients: ", len(patients))
    print("With masses: ", len(with_masses))
    print("No masses: ", len(no_masses))
    """
    SPLIT PATIENTS IN SETS AND LOAD IMAGES
    """
    
    for set_ in ["train","validation","test"]:
        data.add_split(set_)
        for i in range(set_masses_size[set_]):
            patients = with_masses.pop()
            image_files = get_image_files(patients)
            for file in image_files:
                img_iD = data.add_image_file(set_,file)
                data.add_roi_file(img_iD,get_roi(file))
                for mask in get_masks(file):
                    data.add_mask_file(img_iD,mask)
            
        for i in range(set_neg_size[set_]):
            file = no_masses.pop()
            img_iD = data.add_image_file(set_,file)
            data.add_roi_file(img_iD,get_roi(file))
    
    assert len(with_masses) == 0
    assert len(no_masses) == 0
    
    return data

def list_splits():
    splits = glob.glob(os.path.join(split_save_path,"split_*"))
    
    for file in splits:
        split = pkl.load(open(file,"rb"))
        print(split.iD," - ",split.name)
        
def load_split(iD):
    splits = glob.glob(os.path.join(split_save_path,"split_*"))
    
    for file in splits:
        split = pkl.load(open(file,"rb"))
        if split.iD == iD:
            return split
    print("No split with iD: ",iD, "in location: ",split_save_path)
    
def read_img(dataset,iD):
    return np.load(dataset.files_names[iD])
    
Transformation = collections.namedtuple("Transformation","angle mirroring shape dx dy")

def get_random_transformation_parameters(shape=None,dx=None,dy=None):
    angle = np.random.randint(360)           # 0-360
    mirroring = np.random.randint(2)         # 0-1
    
    if shape!=None:
        random_state = np.random.RandomState()
        dx = -1+2*random_state.rand(*shape)
        dy = -1+2*random_state.rand(*shape)
        
        sigma = 20
        alpha = 300
    
        dx = gaussian_filter(dx,sigma,mode="constant",cval=0) * alpha
        dy = gaussian_filter(dy,sigma,mode="constant",cval=0) * alpha
    
    
    return Transformation(angle,  mirroring, shape, dx,dy)

    
def preprocess(image,trans):
    angle = trans.angle
    mirroring = trans.mirroring
    shape = trans.shape
    dx = trans.dx
    dy = trans.dy

    if shape!=None:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)),np.reshape(x+dx, (-1, 1))
    
        image = map_coordinates(image, indices, order=1).reshape(shape)
    
    #imshow(image)
    image = (image*255).astype(np.uint8)
    #imshow(image)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    image = clahe.apply(image)
    rows,cols = image.shape
    
    if mirroring:
        image = cv2.flip(image,0)
        
    M = np.float32([[1,0,trans_const*rows+halfpatch],[0,1,trans_const*rows+halfpatch]])
    image = cv2.warpAffine(image,M,(int(sqrt2*rows)+2*halfpatch,int(sqrt2*rows)+2*halfpatch))
    
    M = cv2.getRotationMatrix2D(((sqrt2*rows)/2+halfpatch,(sqrt2*rows)/2+halfpatch),angle,1)
    dst = cv2.warpAffine(image,M,(int(sqrt2*rows)+2*halfpatch,int(sqrt2*rows)+2*halfpatch))
    
    return dst

def preprocess_mask(mask,trans):
    angle = trans.angle
    mirroring = trans.mirroring
    shape = trans.shape
    dx = trans.dx
    dy = trans.dy

    if shape!=None:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)),np.reshape(x+dx, (-1, 1))
    
        mask = map_coordinates(mask, indices, order=1).reshape(shape)
    
    rows,cols = mask.shape
    mask = mask.astype(np.uint8)
    
    if mirroring:
        mask = cv2.flip(mask,0)
        
    M = np.float32([[1,0,trans_const*rows+halfpatch],[0,1,trans_const*rows+halfpatch]])
    mask = cv2.warpAffine(mask,M,(int(sqrt2*rows)+2*halfpatch,int(sqrt2*rows)+2*halfpatch))
    
    M = cv2.getRotationMatrix2D(((sqrt2*rows)/2+halfpatch,(sqrt2*rows)/2+halfpatch),angle,1)
    mask = cv2.warpAffine(mask,M,(int(sqrt2*rows)+2*halfpatch,int(sqrt2*rows)+2*halfpatch))
    
    return mask
    
def imshow(arr):
    plt.imshow(arr,cmap="gray")
    plt.show()
    
def take_patches(img, points, patch_size):
    patches_arr = np.zeros((points.shape[0],patch_size,patch_size))
    half_size = (patch_size-1)//2
    for i in range(points.shape[0]):
        patch = img[points[i,0]-half_size:points[i,0]+half_size+1,
                    points[i,1]-half_size:points[i,1]+half_size+1]
        patches_arr[i] = patch
    
    return patches_arr    
    

# OBSOLETE!!! TO BE REMOVED IN THE FUTURE
def compute_random_elastic_deformation(img):
    shape = img.shape
    random_state = np.random.RandomState()
    dx = -1+2*random_state.rand(*shape)
    dy = -1+2*random_state.rand(*shape)
    
    sigma = 20
    alpha = 300
    
    dx = gaussian_filter(dx,sigma,mode="constant",cval=0) * alpha
    dy = gaussian_filter(dy,sigma,mode="constant",cval=0) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)),np.reshape(x+dx, (-1, 1))
    
    return map_coordinates(img, indices, order=1).reshape(shape)



dataset = make_dataset()
dataset.make_patches_dataset(1)

"""
image_files = glob.glob(dataset_path+"*")
for file in image_files:
    img = (np.load(file)*255).astype(np.uint8)
    #img = np.load(file)
    imshow(img)
    for i in range(0,img.shape[0],25):
        img[i,:] = img.max()
    for i in range(0,img.shape[1],25):
        img[:,i] = img.max()
        
    for i in range(5):
        trans = get_random_transformation_parameters()
        print(trans)
        
        img2 = data_augment(img,trans)
        print(img2.min(),img2.max())
        imshow(img2)
    input("next?")
"""

"""
img = cv2.imread("/home/edu/example.jpg",0)
#trans = Transformation(angle = 45, scale = 0.6)
t = ut.timer()
trans = get_random_transformation_parameters()
img = data_augment(img,trans)
sa = compute_random_elastic_deformation(img)
imshow(sa)

""" 