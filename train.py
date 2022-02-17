import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import argparse
from XMLDataset import GenSVMFormatParser
from torch.utils.data import DataLoader
import pdb
import yaml
from RzLinear import RzLinear
from tqdm import tqdm

import pyximport
pyximport.install()

import xclib.evaluation.xc_metrics as xc_metrics

class FCN(nn.Module):
    def __init__(self, arch):
        super(FCN, self).__init__()
        layers = []
        lens = [int(i) for i in arch.split('-')]
        for i in range(len(lens) -1):
            layers.append(nn.Linear(lens[i], lens[i+1]))
            layers.append(nn.ReLU())
        layers = layers[:-1]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x
        

class FCNRZ(nn.Module):
    def __init__(self, arch, sizes):
        super(FCNRZ, self).__init__()
        self.layers = []
        self.weights = []
        lens = [int(i) for i in arch.split('-')]
        for i in range(len(lens) -1):
            hashed_weight = nn.Parameter(torch.from_numpy(np.random.uniform(-1/np.sqrt(lens[i+1]), 1/np.sqrt(lens[i+1]), size=sizes[i]).astype(np.float32)))
            rzlinear = RzLinear(lens[i], lens[i+1], 1, hashed_weight, tiled=True)
            self.weights.append(hashed_weight)
            self.layers.append(rzlinear)
            self.layers.append(nn.ReLU())
        self.layers = self.layers[:-1]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        return x

def eval(test_dataloader, model, test_itr):
    actual_labels = []
    predicted_labels = []

    for j,X in tqdm(enumerate(test_dataloader), total=test_itr):
        x = X[0]
        y = X[1]
        actual_labels.append(np.array(y))
        x = x.float().to(dev)
        yhat = model(x)
        predicted_labels.append(np.array(yhat.detach().cpu()))
    print("Metrics .. computing ..")
    A = np.concatenate(actual_labels, axis=0)
    P = np.concatenate(predicted_labels, axis=0)
    acc = xc_metrics.Metrics(true_labels=A)
    args = acc.eval(P, 5)
    print(xc_metrics.format(*args))


def train_epoch(train_dataloader, test_dataloader, model, optimizer, eval_freq, train_itr, test_itr, epoch):
    
    for j,X in tqdm(enumerate(train_dataloader), total=train_itr):
        x = X[0]
        y = X[1]
        optimizer.zero_grad()
        x = x.float().to(dev)
        y = y.float().to(dev)
        yhat = model(x)
        loss = F.binary_cross_entropy_with_logits(yhat, y,  reduction = 'mean')
        loss.backward()
        optimizer.step()
        if (epoch * train_itr + j + 1) % eval_freq == 0:
            print(j, float(loss.detach().cpu()))
            eval(test_dataloader, model, test_itr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--config', action="store", dest="config", type=str, default=None, required=True,
                        help="config to setup the training")
    res = parser.parse_args()
    with open(res.config, "r") as f:
        config = yaml.load(f)

    test_dataset = GenSVMFormatParser(config['data']['test_file'], config["data"])
    train_dataset = GenSVMFormatParser(config['data']['train_file'], config["data"])

    train_samples = train_dataset.__len__()
    test_samples = test_dataset.__len__()
    train_itr = int(train_samples / config['data']['train_batch'])
    test_itr = int(test_samples / config['data']['test_batch'])

    train_dataloader = DataLoader(train_dataset, batch_size=config['data']['train_batch'], shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=config['data']['train_batch'], shuffle=True)
    
    if config['model']['rz']:
        model = FCNRZ(config['model']['arch'], config['model']['sizes'])
    else:
        model = FCN(config['model']['arch'])
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"])
    dev = config["device"]

    if dev != -1:
        model = model.float().to(dev)

    for epoch in range(config["epochs"]):
        train_epoch(train_dataloader, test_dataloader, model, optimizer,  config["eval_freq"], train_itr, test_itr, epoch)
