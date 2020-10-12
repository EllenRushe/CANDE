from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Format: https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.l1 = nn.Linear(28*28, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4= nn.Linear(64, 32)
        self.l5= nn.Linear(32, 3)

    def forward(self, x):
        nn =  F.relu(self.l1(x))
        nn = F.relu(self.l2(nn))
        nn = F.relu(self.l3(nn))
        nn = F.relu(self.l4(nn))
        nn = self.l5(nn)
        return nn

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    cost = nn.CrossEntropyLoss()
    losses = []    
    with tqdm(total=len(train_loader)) as progress_bar:
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.view(data.size(0), -1)
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = cost(output, label)        
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, np.mean(losses)))

def val(model, device, val_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)      
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            data = data.view(data.size(0), -1)
            data = data.to(device)
            target = label.clone()
            softmax = torch.nn.Softmax(dim=1)
            output = softmax(model(data))
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
    
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(targets, preds)
    return acc

def test(model, device, test_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)    
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as progress_bar:
            for batch_idx, (data, label) in enumerate(test_loader):
                data = data.view(data.size(0), -1)
                data = data.to(device)
                target = label.clone()
                softmax = torch.nn.Softmax(dim=1)
                output = softmax(model(data))
                preds.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
                progress_bar.update(1)

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(targets, preds)
    return acc
