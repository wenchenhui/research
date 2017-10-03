
import data.cbis_standart_format as cbis_standart_format
import simple_data_processor
import train_loop
import pickle as pkl
import os
import configs_funcs as cfunc




"""
EASY TO USE EXPERIMENT
SAVE ONE COPY AS TEMPLATE
"""

dataset = "inbreast"
inv_scale = 12
clahe = True


first_dataset_name = dataset
use_elastic_deform= True
no_transformations = 40

experiment_name = "inbreast_easy_test_same_exp"


batch_norm = False
dropout = True


pre_data_folder = cfunc.get_path_predata(dataset,inv_scale,clahe)

if not os.path.isdir(pre_data_folder):
    cfunc.create_standart_predata(dataset,inv_scale,clahe,debug=False)
    
if False:
    data_folder = cfunc.get_path_data1(first_dataset_name, use_elastic_deform, no_transformations)

    if not os.path.isdir(data_folder):
        dataset = cfunc.create_first_dataset(pre_data_folder, first_dataset_name , use_elastic_deform, no_transformations)
    else:
        dataset = cfunc.load_dataset(first_dataset_name, use_elastic_deform, no_transformations)
    
if False:
    experiment_name = "Batch_Dropout_exp_"+str(batch_norm)+"_"+str(dropout)
    cfunc.train_model1(data_folder,experiment_name)
    cfunc.test_model1(dataset,experiment_name)




#cfunc.create_second_dataset(data_folder,experiment_name)
#cfunc.train_model2(experiment_name)
#cfunc.test_model2(dataset,experiment_name)







"""
    TEST IF CLAHE MAKES RESULTS BETTER IN CBIS-DDSM
"""

if False:
    
    
        
    experiment_name = "elastic_deforms_"+str(True)
    data_location = "/home/eduardo/Results/"+experiment_name+"/second_dataset/"
    
    train_loop.train_loop("elastic_deforms_"+str(True),2,data_location)    
    
    
    #train_loop.create_second_dataset("elastic_deforms_"+str(True),"/home/eduardo/dataset_elastic_trans_True/")
    
    input("CONTINUE?")
    
    input("SURE?")
    
    inv_scale = 24
    scale = 1/inv_scale
    
    for use_elastic_deform in [True]:
        clahe = True
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
        dst_location = "/home/eduardo/dataset_elastic_trans_"+str(clahe)+"/"
        
        set_masses_size = {"train":491,"validation":100,"test":100}
        set_neg_size = {"train":0,"validation":0,"test":0}
        
        if not os.path.isfile(dst_location+"/dataset_test"):    
            dataset = simple_data_processor.make_dataset(set_masses_size, set_neg_size, src_location, dst_location)
            
            halfpatch= 18
        
            no_transformations = 40
            use_rotations = True
            use_mirroring = True
            #use_elastic_deform = True
            
            debug = False
            dataset.make_patches_dataset(no_transformations,use_rotations,use_mirroring,use_elastic_deform,debug)
            dataset.save(dst_location)
        else:
            print("Loading dataset")
            dataset = pkl.load(open(dst_location+"/dataset_test","rb"))
        
        
        train_loop.train_loop("elastic_deforms_"+str(use_elastic_deform),1,dst_location)    
        all_suspicions = train_loop.test_model(1,"elastic_deforms_"+str(use_elastic_deform),dataset,sigma=0.8,num_dets=40)
    



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
