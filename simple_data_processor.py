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
import funcs.utils as ut

# CONTANTS
sqrt2 = np.sqrt(2)
trans_const = (sqrt2-1)/2




"""
_______________________________________________________________________________
_______________________________DAYDREAMING_____________________________________
MAIN FUNCTIONS (DATASET OBJECT) AND MAKE DATASET FUNCTION
"""

class dataset():
    
    def __init__(self, name, dataset):
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
            raise("Duplicate roi iD (",iD,") in file: ", file)
    
    def make_patches_dataset(self, no_transformations, use_rotations,
                             use_mirroring, use_elastic_deform, debug):
        
        global _use_rotations,_use_mirroring,_use_elastic_deform, _debug
        _use_rotations = use_rotations
        _use_mirroring = use_mirroring
        _use_elastic_deform = use_elastic_deform
        _debug = debug
        #COUNTER USED TO NAME PATCHES UNIQUELY
        counter = 0
        
        
        #FOR EACH SPLIT        
        for split in self.splits.keys():
            print("Doing:",split)
            bar = ut.progress_bar(len(self.splits[split]))
            #FOR EACH IMAGE OF CURRENT SPLIT
            for img_iD in self.splits[split]:
                bar.tick()                
                # LOAD IMAGE, MASKS AND ROI. CREATE FULL MASK (SUM OF MASKS)
                img_ori = np.load(self.files_names[img_iD])
                masks_ori = _load_file_array(self.masks[img_iD])
                roi_ori = np.load(self.rois[img_iD])
                full_mask_ori = _sum_masks(masks_ori)
                
                # FOR no_transformations TIMES
                for i in range(no_transformations):
                    
                    # GET A RANDOM TRANFORMATION PARAMETRIZATION
                    trans = _get_random_transformation_parameters()    
                    
                    #APPLY TO IMAGE, MASKS AND ROI
                    img = _preprocess(img_ori,trans)
                    roi = _preprocess_mask(roi_ori,trans)
                    full_mask = _preprocess_mask(full_mask_ori,trans)

                    # CREATE AND SAVE NEGATIVE PATCHES
                    patches = _take_negative_patches(img,roi,full_mask)
                    counter = _save_patches(split,"negative",patches,counter)
                    
                    # CREATE AND SAVE POSITIVE PATCHES (ONE FOR EACH MASK)
                    for mask in masks_ori:
                        mask = _preprocess_mask(mask,trans)
                        patches = _take_positive_patches(img,mask)
                        counter = _save_patches(split,"positive",patches,counter)
                        
    def save(self,path):
        pkl.dump(self,open(path+"/dataset_"+self.name,"wb"))


def make_dataset(set_masses_size, set_neg_size, src_loc, dst_loc):
    global _src_location, _dst_location
    _src_location = src_loc
    _dst_location = dst_loc
    
    print("CREATING A NEW DATASET OBJECT")
    data = dataset("test","CBIS-DDSM")
    
    # GET LIST OF PATIENTS
    patients = _get_patients()
    
    # SEPARATE PATIENTS WITH MASSES AND WITHOUT MASSES
    with_masses, no_masses = [],[]
    for patient in patients:
        if _has_mass(patient):
            with_masses.append(patient)
        else:
            no_masses.append(patient)
    
    # SHUFFLE SO WE CAN OBTAIN DIFFERENT DATA DISTRIBUTION EACH TIME
    random.shuffle(with_masses)
    random.shuffle(no_masses)
    
    # PRINT SOME INFORMATION TO THE USER
    print("\t Patients with masses: ", len(with_masses))
    print("\t Patients without masses: ", len(no_masses))
    print("\t Total: ", len(patients))

    
    
    # FOR EACH SPLIT
    for set_ in ["train","validation","test"]:
        
        # ADDS A NEW SPLIT IN THE DATASET (TRAIN/VAL/TEST)
        data.add_split(set_)
        
        # POP PATIENTS FROM LIST UNTIL WE HAVE THE NUMBER REQUESTED BY THE USER
        for i in range(set_masses_size[set_]):
            
            patient = with_masses.pop()
            image_files = _get_image_files(patient)
            
            # ADD THE IMAGES OF THE CURRENT PATIENT AND THEIR MASKS TO 
            #THE DATASET
            for file in image_files:
                img_iD = data.add_image_file(set_,file)
                data.add_roi_file(img_iD,_get_roi(file))
                
                for mask in _get_masks(file):
                    data.add_mask_file(img_iD,mask)
        
        # POP PATIENTS FROM LIST UNTIL WE HAVE THE NUMBER REQUESTED BY THE USER
        for i in range(set_neg_size[set_]):
            file = no_masses.pop()
            img_iD = data.add_image_file(set_,file)
            data.add_roi_file(img_iD,_get_roi(file))
    
    # ASSERT NO PATIENT IS LEFT BEHIND
    assert len(with_masses) == 0
    assert len(no_masses) == 0
    
    print("DATASET CREATED")
    return data
        
    
"""
_______________________________________________________________________________
_______________________________DAYDREAMING_____________________________________
FUNCTIONS TO DEAL WITH PATCHES
"""    
_halfpatch=18
patches = np.zeros((10000,_halfpatch*2,_halfpatch*2))

def _take_negative_patches(img,roi,full_mask):
    x = np.arange(0,img.shape[0],_halfpatch*2) 
    y = np.arange(0,img.shape[1],_halfpatch*2)
    
    #plt.imshow(img)
    #plt.show()
    counter = 0
    #print(x)
    #print(y)
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            if roi[x[i],y[j]] == 1:
                #print(img.shape)                
                #print(":::",x[i],y[j])
                if full_mask[x[i]-_halfpatch:x[i]+_halfpatch,y[j]-_halfpatch:y[j]+_halfpatch].sum()==0:
                    patches[counter] = img[x[i]-_halfpatch:x[i]+_halfpatch,y[j]-_halfpatch:y[j]+_halfpatch]
                    counter+=1

    return patches[0:counter].copy()

def _take_positive_patches(img,mask):
    centroid = center_of_mass(mask)
    centroid = np.round(centroid).astype(int)
    patch = img[centroid[0]-_halfpatch:centroid[0]+_halfpatch,centroid[1]-_halfpatch:centroid[1]+_halfpatch]
    return [patch]



# CURRENTLY NOT BEING USED
# MIGHT BE IMPORTANT IN THE FUTURE    
def _take_patches(img, points, patch_size):
    patches_arr = np.zeros((points.shape[0],patch_size,patch_size))
    half_size = (patch_size-1)//2
    for i in range(points.shape[0]):
        patch = img[points[i,0]-half_size:points[i,0]+half_size+1,
                    points[i,1]-half_size:points[i,1]+half_size+1]
        patches_arr[i] = patch
    
    return patches_arr



def _save_patches(split,folder,list_of_patches,counter):
    directory = _dst_location+"/"+split+"/"+folder+"/"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    for p in list_of_patches:
        if _debug:
            imsave(directory+str(counter)+".png",p)
        else:
            np.save(directory+str(counter),p)
        counter+=1

    return counter 





"""
_______________________________________________________________________________
___________________________________AZEALIA_____________________________________
FUNCTIONS TO LIST DATA
"""
def _get_patients():
    patients = set()
    files = glob.glob(_src_location+"*")
    for file in files:
        pat = os.path.basename(file).split("_")[0]
        patients.add(pat)
    
    return list(patients)
    
def _has_mass(patient):
    images = _get_image_files(patient)
    for image in images:
        if len(_get_masks(image)) != 0:
            return True
    return False
    
def _get_image_files(patient):
    return glob.glob(os.path.join(_src_location,patient+"_img*"))
    
def _get_roi(image_file):
    return glob.glob(image_file.replace("img","roi").split(".")[0]+"*")[0]
    
def _get_masks(image_file):
   return glob.glob(image_file.replace("img","mask").split(".")[0]+"_*")
    
    
"""
_______________________________________________________________________________
___________________________________AZEALIA_____________________________________
FUNCTIONS TO LOAD DATA. SIMPLE PROCESSING METHODS
"""

def _read_img(dataset,iD):
    return np.load(dataset.files_names[iD])
    
def _load_file_array(list_of_files):
    result = []
    for file in list_of_files:
        result.append(np.load(file))
    return result
    
def _sum_masks(mask_list):
    total = np.zeros(mask_list[0].shape)
    for i in range(0,len(mask_list)):
        total+=mask_list[i]
    total = np.ceil(total)
    total = np.clip(total,0,1)
    return total
    
def _imshow(arr):
    plt.imshow(arr,cmap="gray")
    plt.show()
    
    
"""
_______________________________________________________________________________
________________________________DAYDREAMING____________________________________
DATA AUGMENTATION FUNCTIONS
    CREATES RANDOM TRANSFORMATIONS OF IMAGES/MASKS/ROIS
"""

Transformation = collections.namedtuple("Transformation","angle mirroring shape dx dy")

def _get_random_transformation_parameters(shape=None,dx=None,dy=None):
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

    
def _preprocess(image,trans):
    
    angle = trans.angle
    mirroring = trans.mirroring
    shape = trans.shape
    dx = trans.dx
    dy = trans.dy
    
    if _use_elastic_deform:
        if shape!=None:
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y+dy, (-1, 1)),np.reshape(x+dx, (-1, 1))
        
            image = map_coordinates(image, indices, order=1).reshape(shape)
    
    rows,cols = image.shape
    
    if mirroring and _use_mirroring:
        image = cv2.flip(image,0)
      
    if _use_rotations:
        added = int(0.8*rows)+_halfpatch
        
        image = cv2.copyMakeBorder(image,added,added,added,added,cv2.BORDER_REFLECT_101)
        
        M = cv2.getRotationMatrix2D((image.shape[0]/2,image.shape[1]/2),angle,1)
        image = cv2.warpAffine(image,M,(image.shape[0],image.shape[1]))
    
    return image

def _preprocess_mask(mask,trans):
    angle = trans.angle
    mirroring = trans.mirroring
    shape = trans.shape
    dx = trans.dx
    dy = trans.dy

    mask = mask.astype(float)
    
    if _use_elastic_deform:
        if shape!=None:
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y+dy, (-1, 1)),np.reshape(x+dx, (-1, 1))
        
            mask = map_coordinates(mask, indices, order=1).reshape(shape)
    
    rows,cols = mask.shape
    
    if mirroring and _use_mirroring:
        mask = cv2.flip(mask,0)
     
     
    if _use_rotations:
        added = int(0.8*rows)+_halfpatch
        
        mask = cv2.copyMakeBorder(mask,added,added,added,added,cv2.BORDER_CONSTANT,value=0)
        
        M = cv2.getRotationMatrix2D((mask.shape[0]/2,mask.shape[1]/2),angle,1)
        mask = cv2.warpAffine(mask,M,(mask.shape[0],mask.shape[1]))
    
    return mask
    
"""
def _list_splits():
    splits = glob.glob(os.path.join(split_save_path,"split_*"))
    
    for file in splits:
        split = pkl.load(open(file,"rb"))
        print(split.iD," - ",split.name)
        
def _load_split(iD):
    splits = glob.glob(os.path.join(split_save_path,"split_*"))
    
    for file in splits:
        split = pkl.load(open(file,"rb"))
        if split.iD == iD:
            return split
    print("No split with iD: ",iD, "in location: ",split_save_path)
    
    
"""
