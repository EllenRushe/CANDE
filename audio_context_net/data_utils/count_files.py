import logging
logging.basicConfig(level = logging.DEBUG, filename = "baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
########################################################################


########################################################################
# import default python-library
########################################################################
import pickle
import os
import sys
import glob
import pandas as pd
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
numpy.random.seed(0) # set seed
import librosa
import librosa.core
import librosa.feature

# from import
from tqdm import tqdm

def dataset_generator(target_dir, 
                      normal_dir_name = "normal", 
                      abnormal_dir_name = "abnormal", 
                      ext = "wav"):
    """
    target_dir : str
        base directory path of the dataset
    normal_dir_name : str (default="normal")
        directory name the normal data located in
    abnormal_dir_name : str (default="abnormal")
        directory name the abnormal data located in
    ext : str (default="wav")
        filename extension of audio files 

    return : 
        train_data : numpy.array( numpy.array( float ) )
            training dataset
            * dataset.shape = (total_dataset_size, fearture_vector_length)
        train_files : list [ str ]
            file list for training
        train_labels : list [ boolean ] 
            label info. list for training
            * normal/abnnormal = 0/1
        eval_files : list [ str ]
            file list for evaluation
        eval_labels : list [ boolean ] 
            label info. list for evaluation
            * normal/abnnormal = 0/1
    """
    logger.info("target_dir : {}".format(target_dir))

    # 01 normal list generate
    normal_files = sorted(glob.glob("{dir}/{normal_dir_name}/*.{ext}".format(
                   dir = target_dir, 
                   normal_dir_name = normal_dir_name, 
                   ext = ext))
                   )
    normal_labels = numpy.zeros(len(normal_files))
    if len(normal_files) == 0: logger.exception(f'{"no_wav_data!!"}')

    # 02 abnormal list generate
    abnormal_files = sorted(glob.glob( "{dir}/{abnormal_dir_name}/*.{ext}".format(dir = target_dir, abnormal_dir_name = abnormal_dir_name, ext = ext)))
    abnormal_labels = numpy.ones(len(abnormal_files))
    if len(abnormal_files) == 0: logger.exception(f'{"no_wav_data!!"}')
    # 03 separate train & eval
    train_files = normal_files[len(abnormal_files):]
    train_labels = normal_labels[len(abnormal_files):]
    '''MODIFICATION'''
    num_val = round(len(train_files)*(10)/100)
    val_files = normal_files[:num_val]
    val_labels = normal_labels[:num_val]
    '''-----------'''
    eval_files = numpy.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
    eval_labels = numpy.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)
    
    logger.info("train_file num : {num}".format(num = len(train_files)))
    '''MODIFICATION'''
    logger.info("val_file num : {num}".format(num = len(val_files)))    
    '''-----------'''
    logger.info("eval_file  num : {num}".format(num = len(eval_files)))
    '''MODIFICATION'''
    return train_files, train_labels, val_files, val_labels, eval_files, eval_labels    

base_dir = "/home/people/12329741/scratch/machine_sound/mimii_baseline/dataset"
dirs = sorted(glob.glob("{base}/*/*/*".format(base = base_dir)))
data = []
for num, target_dir in enumerate(dirs):
        train_files, train_labels, val_files, val_labels, eval_files, eval_labels = dataset_generator(target_dir)
        db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
        machine_type = os.path.split(os.path.split(target_dir)[0])[1]
        machine_id = os.path.split(target_dir)[1]
        print("Target dir:", target_dir)
        print("Train files:", len(train_files))
        print("Val files:", len(train_files))
        print("Test files:", len(train_files))
        data.append( {"machine type": machine_type, 
                "machine id": machine_id, 
                "SNR": db,
                "train": len(train_files),
                "val": len(val_files), 
                "test total": len(eval_files),
                "test normal": len(eval_files[eval_labels==0]), 
                "test abnormal": len(eval_files[eval_labels==1])
                })
df = pd.DataFrame(data)
df.to_csv("mimii_file_counts.csv")        

