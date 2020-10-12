import os
import torch
import pickle
import numpy as np
from torch.utils.data.dataset import Dataset


class MNISTDataset(Dataset):
    def __init__(self, data_dir, X_filename, y_filename, 
        context_filename=None, flatten = False, context_embeddings_dir = None, context_label=None):
        """
        data_dir (str): Path to data containing data and labels. 
        X_filename (str): Name of file containing input data. 
        y_filename (str): Name of file containing labels.
        """
        # Context should really be in the data. 
        self.data = torch.from_numpy(
            np.load(os.path.join(data_dir, X_filename)).astype(np.float32))
        if flatten:
            self.data = self.data.view(self.data.shape[0], -1)

        self.labels =torch.from_numpy(
            np.load(os.path.join(data_dir, y_filename)).astype(np.float32)
            )
        
        self.context_embeddings_dir = context_embeddings_dir
        # Initialise embedding size as None in case there isn't one. 
        self.context_embedding_size = None
        self.context_filename = context_filename
        self.context_label = context_label         
        if (self.context_filename  is not None) or (self.context_label is not None):
            if self.context_filename  is not None:
                self.contexts_int = np.load(os.path.join(data_dir, context_filename)).astype(np.int32)
            elif self.context_label is not None:
                self.contexts_int = np.full(len(self.data), self.context_label)   
            if context_embeddings_dir  is not None:
                self.embeddings = torch.from_numpy(
                    np.load(context_embeddings_dir).astype(np.float32)
                    )
                self.context_embedding_size = self.embeddings.shape[-1]

            else:

                contexts = np.zeros((self.contexts_int.shape[0], 3))
                contexts[np.arange(self.contexts_int.shape[0]), self.contexts_int] = 1
                contexts = contexts.astype(np.float32)
                self.contexts = torch.from_numpy(contexts )

    def get_embedding_size(self):
        return self.context_embedding_size           

    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]

        if (self.context_filename is not None) or (self.context_label is not None):
            if self.context_embeddings_dir  is not None:
                context  = self.embeddings[self.contexts_int[index]]
            else:
                context = self.contexts[index]
            return X, y.long(), context
        # Targets are integers,and are one-hot encoded within the loss function automatically. 
        return X, y.long()

    def __len__(self):
        return len(self.data)
