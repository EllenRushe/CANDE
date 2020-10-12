from __future__ import print_function
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data_utils.Datasets import MNISTDataset
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
# Format: https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py

class FiLM(nn.Module):
    ''''
        Form: https://github.com/ethanjperez/film/blob/master/vr/models/filmed_net.py
    '''
    def forward(self, layer, gammas, betas):
        return (gammas * layer) + betas

class FiLMBlock(nn.Module):
    def __init__(self, in_size, out_size, in_betas, in_gammas):
        super(FiLMBlock, self).__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.betas_fc = nn.Linear(in_betas, out_size)
        self.gammas_fc = nn.Linear(in_gammas, out_size)
        self.film = FiLM()

    def forward(self, x, gs, bs):
        x = self.fc(x)
        gammas = self.gammas_fc(gs)
        betas =  self.betas_fc(bs)
        return F.relu(self.film(x, gammas, betas))

'''
class FiLMBlock(nn.Module):
    def __init__(self, in_size, out_size, in_betas, in_gammas):
        super(FiLMBlock, self).__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.embeddings_gamma = nn.Embedding(in_gammas, out_size)
        self.embeddings_beta = nn.Embedding(in_betas, out_size)
        # Initialise scaler to ~1 
        self.embeddings_gamma.weight.data.normal_(1, 0.02)
        # Set shifter to 0.
        self.embeddings_beta.weight.data.zero_()
        self.film = FiLM()

    def forward(self, x, embed_idx):
        x = F.relu(self.fc(x))
        gammas = self.embeddings_gamma(embed_idx)
        betas =  self.embeddings_beta(embed_idx)
        return F.relu(self.film(x, gammas, betas))
'''


class net(nn.Module):
    def __init__(self, **kwargs):
        super(net, self).__init__()
        embed_size = 3
        self.en_1 = nn.Linear(28*28, 128)
        self.en_2 = FiLMBlock(128, 64, embed_size, embed_size)
        self.en_3 = FiLMBlock(64, 32, embed_size, embed_size)
        self.de_1= FiLMBlock(32, 64, embed_size, embed_size)
        self.de_2 = FiLMBlock(64, 128, embed_size, embed_size)
        self.de_3 = nn.Linear(128, 28*28)


    def forward(self, x, context):
        encoder  = F.relu(self.en_1(x))
        encoder = self.en_2(encoder, context, context)
        encoder = self.en_3(encoder, context, context)
        decoder = self.de_1(encoder, context, context)
        decoder = self.de_2(decoder, context, context)
        decoder = self.de_3(decoder)
        return decoder

def train(model, device, train_loader, optimizer, epoch, cost):
    model.train()
    losses = []
    with tqdm(total=len(train_loader)) as progress_bar:
        for batch_idx, (data, _, context) in enumerate(train_loader):
            data = data.view(data.size(0), -1)
            data = data.to(device)
            context = context.to(device)
            optimizer.zero_grad()
            output = model(data, context)
            loss = cost(output, data)        
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            progress_bar.update(1)
    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, np.mean(losses)))


def val(model, device, val_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)      
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as progress_bar:
            for batch_idx,(data,_, context) in enumerate(val_loader):
                # Flatten data (for fully connected network)
                data = data.view(data.size(0), -1)
                data = data.to(device)
                context = context.to(device)
                target = data.clone()
                output = model(data, context)
                preds.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
                progress_bar.update(1)

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    error = np.mean((preds - targets)**2, axis=(1))
    mean_error = np.mean(error)
    return mean_error

def test(model, device, test_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)         
    model.eval()
    preds = []
    targets = []
    labels = []
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as progress_bar:
            for batch_idx, (data, label, context) in enumerate(test_loader):
                data = data.view(data.size(0), -1)
                data = data.to(device)
                context = context.to(device)
                target = data.clone()
                y = label.clone()
                output = model(data, context)
                preds.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
                labels.append(y)
                progress_bar.update(1)    

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    labels = np.concatenate(labels)
    return preds, targets, labels

        