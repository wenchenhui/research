
import data.cbis_standart_format as cbis_standart_format
import simple_data_processor
import train_loop
import pickle as pkl
import os

"""
    TEST IF CLAHE MAKES RESULTS BETTER IN CBIS-DDSM
"""
inv_scale = 24
scale = 1/inv_scale

for clahe in [False,True]:

    src_location = "/media/eduardo/TOSHIBA EXT/"
    folder_images = "raw CBIS mass/DOI/"
    folder_masks = "raw CBIS mass rois/DOI/"
    dst_location = "/home/eduardo/pre_dataset_clahe_"+str(clahe)+"_"+str(inv_scale)+"/"

    debug = False
    if not os.path.isdir(dst_location):
        cbis_standart_format.make_CBIS_standart_format(scale,clahe,src_location,folder_images,folder_masks,dst_location,debug)
    else:
        print("PRE-DATASET ALREADY MADE")
      
    src_location = dst_location
    dst_location = "/home/eduardo/dataset_clahe_"+str(clahe)+"/"
    
    set_masses_size = {"train":491,"validation":100,"test":100}
    set_neg_size = {"train":0,"validation":0,"test":0}
    
    if not os.path.isfile(dst_location+"/dataset_test"):    
        dataset = simple_data_processor.make_dataset(set_masses_size, set_neg_size, src_location, dst_location)
        
        halfpatch= 18
    
        no_transformations = 20
        use_rotations = True
        use_mirroring = True
        use_elastic_deform = False
        
        debug = False
        dataset.make_patches_dataset(no_transformations,use_rotations,use_mirroring,use_elastic_deform,debug)
        dataset.save(dst_location)
    else:
        print("Loading dataset")
        dataset = pkl.load(open(dst_location+"/dataset_test","rb"))
    
    train_loop.train_loop("clahe_test_"+str(clahe),1,dst_location)    
    all_suspicions = train_loop.test_model(1,"clahe_"+str(clahe),dataset,sigma=0.8,num_dets=40)
    



"""
    WHOLE PROCESS OF LEARNING A PATCH MODEL FROM FULL IMAGES
"""


if False:
    
    """ CBIS_STANDART_FORMAT """
    scale = 1/24
    clahe = False
    
    src_location = "/media/eduardo/TOSHIBA EXT/"
    folder_images = "raw CBIS mass/DOI/"
    folder_masks = "raw CBIS mass rois/DOI/"
    dst_location = "/media/eduardo/TOSHIBA EXT/tmp_testdataset/"
    
    debug = False
    
    
    #cbis_standart_format.make_CBIS_standart_format(scale,clahe,src_location,folder_images,folder_masks,dst_location,debug)
    
    """ Simple Data Processor """
    
    
    src_location = dst_location
    dst_location = "/home/eduardo/tmp_testdataset2ndpart"
    
    set_masses_size = {"train":491,"validation":100,"test":100}
    set_neg_size = {"train":0,"validation":0,"test":0}
    
    #dataset = simple_data_processor.make_dataset(set_masses_size, set_neg_size, src_location, dst_location)
    
    # PATCHES
    
    halfpatch= 18
    
    no_transformations = 5
    use_rotations = True
    use_mirroring = True
    use_elastic_deform = False
    
    debug = False
    #dataset.make_patches_dataset(no_transformations,use_rotations,use_mirroring,use_elastic_deform,debug)
    #dataset.save(dst_location)
    dataset = pkl.load(open(dst_location+"/dataset_test","rb"))
    #train_loop.train_loop("name5",1,dst_location)
    
    all_suspicions = train_loop.test_model(1,"name5",dataset)
