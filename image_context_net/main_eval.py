from __future__ import print_function
import os
import time
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, auc

import models 
from data_utils import Datasets
from data_utils.params import Params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "val_json", type=str, help="Directory of validation json file which indictates the best epoch.")
    parser.add_argument(
            "iter", type=int, default=1, help="Evaluation iteration.")
    args = parser.parse_args()

    with open(args.val_json) as json_file:  
        model_params  = json.load(json_file)  
    params = Params("hparams.yaml", model_params["model"])
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=params.gpu_vis_dev
    use_gpu= torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    log_dir = os.path.join(params.log_dir, "test_eval")
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    if not os.path.exists(params.output_dir): os.makedirs(params.output_dir)
    model_module = __import__('.'.join(['models', params.model_name]),  fromlist=['object'])
    Dataset = getattr(Datasets, params.dataset_class)
    
    #model = model_module.net(embedding_size=test_data.get_embedding_size()).to(device)
    #model = nn.DataParallel(model).to(device) 
   


    iter_checkpoint = os.path.join(
        params.checkpoint_dir,
        "checkpoint_{}_{}_iter_{}_epoch_{}".format(model_params["model"], model_params["context"], args.iter,  model_params["best_val_epoch"])
    )
    results = {}

    results_log = []
    for context_i in [0 ,1 ,2]:
        results_dir = os.path.join("results", model_params["model"], model_params["context"]) 
        if not os.path.exists(results_dir): os.makedirs(results_dir)
        if model_params["context"]:
            # Skip these files if they aren't in the context being evaluated. 
            if context_i != int(model_params["context"]):
                continue
            # There will be no context label if no context filename was used for training. 
        if params.context_filename == None:
            context_label = None
        else:
            context_label =context_i  

        test_data_dir = os.path.join(params.eval_dir, "test", str(context_i))
        
        test_data = Dataset(
                test_data_dir, 
                'X.npy', 
                'y.npy',
                context_filename=None,
                context_label=context_label,
                context_embeddings_dir=params.context_embedding_file
                )        
        test_loader = DataLoader(
                test_data, 
                batch_size=params.batch_size, 
                shuffle=False, 
                num_workers=1
                )
        model = model_module.net(embedding_size=test_data.get_embedding_size()).to(device)
        model = nn.DataParallel(model).to(device)
        test = model_module.test
        preds, targets, labels = test(
                model, device, test_loader, checkpoint=iter_checkpoint
        )

        error = np.mean((preds - targets)**2, axis=(1))
        fpr, tpr, thresholds = roc_curve(labels, error)
        auc_score =  auc(fpr, tpr)
        results["{}".format(context_i)] = {"AUC": float(auc_score)}
        results_file_name = "results_{}.json".format(iter_checkpoint.split('/')[-1])
        with open(os.path.join(results_dir, results_file_name), "w") as f:
            json.dump(results, f)                
                        
        np.save(os.path.join(
                        params.output_dir, 
                        "preds_test_{}_{}_iter_{}_{}".format(
                                model_params["model"], 
                                context_i,
                                args.iter, 
                                time.strftime("%d%m%y_%H%M%S"))), preds)
        results_log.append(float(auc_score))

    
    logs = {"model": model_params["model"], 
            "checkpoint_name": iter_checkpoint, 
            "val_json": args.val_json,
            "auc_scores": results_log
        }

    with open(
            os.path.join(
                    log_dir, 
                    "test_{}_{}_iter_{}_{}.json".format(
                            model_params["model"], 
                            model_params["context"],
                            args.iter, 
                            time.strftime("%d%m%y_%H%M%S"))), 'w') as f:
            json.dump(logs, f)

if __name__ == '__main__':

        main()
