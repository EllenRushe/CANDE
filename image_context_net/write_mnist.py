import os
import torch
import shutil
import numpy as np
# shuffle() is basically resample() with replace=False
from sklearn.utils import shuffle
from matplotlib.pyplot import imshow
from torchvision import datasets

from data_utils.data_utils import *


def write_semisupervised_all_context_MNIST(file_path, context_dict):
    train_data = datasets.MNIST('./data', train=True, download=True
        )
    test_data = datasets.MNIST('./data', train=False, 
            )

    train_X, train_y = train_data.data.numpy(), train_data.train_labels.numpy()
    test_X, test_y = test_data.data.numpy(), test_data.test_labels.numpy()

    train_X, train_y = shuffle(train_X, train_y, random_state=1)
    test_X, test_y = shuffle(test_X, test_y, random_state=1)

    train_X = train_X / 255 
    test_X = test_X / 255
    print("ORIGINAL TRAIN SIZE",train_X.shape)
    print("ORIGINAL TEST SIZE",test_X.shape)

    val_split = int(train_X.shape[0]*.1)


    val_X, val_y = train_X[:val_split], train_y[:val_split]
    train_X, train_y = train_X[val_split:], train_y[val_split:]
    
    train_y_unique, train_y_count = np.unique(train_y, return_counts=True)
    val_y_unique, val_y_count = np.unique(val_y, return_counts=True)
    test_y_unique, test_y_count = np.unique(test_y, return_counts=True)
    print("Train Unique:\t", train_y_unique)
    print("Train Freq.:\t", train_y_count)
    print("Val Unique:\t", val_y_unique)
    print("Val Freq.:\t", val_y_count)    
    print("Test Unique:\t", test_y_unique)
    print("Test Freq.:\t", test_y_count)

    set_pairs = [[train_X, train_y], [val_X, val_y], [test_X, test_y]]
    set_dirs = ["train", "val", "test"]
    for set_i in range(len(set_pairs)):
        # Only keep anomalous labels in the test set. 
        if set_dirs[set_i] == 'test':
            write_data_all_contexts(set_pairs[set_i][0], 
                                   set_pairs[set_i][1], 
                                   context_dict, 
                                   "normal", 
                                   "anom", 
                                   keep_anom=True, 
                                   filepath=os.path.join(file_path, set_dirs[set_i]), 
                                   seed=1
                                    )
        else:
            write_data_all_contexts(set_pairs[set_i][0], 
                       set_pairs[set_i][1], 
                       context_dict, 
                       "normal", 
                       "anom", 
                       keep_anom=False, 
                       filepath=os.path.join(file_path, set_dirs[set_i]), 
                       seed=1
                        )


def write_semisupervised_sep_context_MNIST(file_path, context_dict):
    train_data = datasets.MNIST('./data', train=True, download=True
        )
    test_data = datasets.MNIST('./data', train=False, 
            )

    train_X, train_y = train_data.data.numpy(), train_data.train_labels.numpy()
    test_X, test_y = test_data.data.numpy(), test_data.test_labels.numpy()

    train_X, train_y = shuffle(train_X, train_y, random_state=1)
    test_X, test_y = shuffle(test_X, test_y, random_state=1)
    print("ORIGINAL TRAIN SIZE",train_X.shape)
    print("ORIGINAL TEST SIZE",test_X.shape)
    train_X = train_X / 255 
    test_X = test_X / 255


    val_split = int(train_X.shape[0]*.1)


    val_X, val_y = train_X[:val_split], train_y[:val_split]
    train_X, train_y = train_X[val_split:], train_y[val_split:]
    
    train_y_unique, train_y_count = np.unique(train_y, return_counts=True)
    val_y_unique, val_y_count = np.unique(val_y, return_counts=True)
    test_y_unique, test_y_count = np.unique(test_y, return_counts=True)
    print("Train Unique:\t", train_y_unique)
    print("Train Freq.:\t", train_y_count)
    print("Val Unique:\t", val_y_unique)
    print("Val Freq.:\t", val_y_count)    
    print("Test Unique:\t", test_y_unique)
    print("Test Freq.:\t", test_y_count)

    set_pairs = [[train_X, train_y], [val_X, val_y], [test_X, test_y]]
    set_dirs = ["train", "val", "test"]
    for set_i in range(len(set_pairs)):
        # Only keep anomalous labels in the test set. 
        if set_dirs[set_i] == 'test':
            write_data_sep_context(set_pairs[set_i][0], 
                                   set_pairs[set_i][1], 
                                   context_dict, 
                                   "normal", 
                                   "anom", 
                                   keep_anom=True, 
                                   filepath=os.path.join(file_path, set_dirs[set_i]), 
                                   seed=1)
        else:
            write_data_sep_context(set_pairs[set_i][0], 
                       set_pairs[set_i][1], 
                       context_dict, 
                       "normal", 
                       "anom", 
                       keep_anom=False, 
                       filepath=os.path.join(file_path, set_dirs[set_i]), 
                       seed=1)     
def main():

  context_dict = {  
          0: 
              {
                  'normal':{0,1,2}, 
                  'anom': {3}
              }, 
          1: 
              {
                  'normal':{3,4,5}, 
                  'anom' : {6}
              }, 
          2: 
              {
                  'normal':{6,7,8}, 
                  'anom':{0}
              }

          }


  semisupervised_sep_mnist_dir = "mnist/semisupervised_sep"
  write_semisupervised_sep_context_MNIST(semisupervised_sep_mnist_dir, context_dict)


  semisupervised_all_mnist_dir = "mnist/semisupervised_all"
  write_semisupervised_all_context_MNIST(semisupervised_all_mnist_dir, context_dict)      



if __name__ == '__main__':
    main()
