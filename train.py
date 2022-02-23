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
from hashedEmbeddingBag import HashedEmbeddingBag
from tqdm import tqdm
import tempfile

import pyximport
pyximport.install()

import xclib.evaluation.xc_metrics as xc_metrics

def sparse_collate_function(pointlist, pad_token=0):
    labels = []
    data = []
    weights = []
    maxlen = 0
    for point in pointlist:
        labels.append(point[1])
        data.append(point[0][0])
        weights.append(point[0][1])
        maxlen = max(maxlen, len(point[0][0]))
    t_label = torch.from_numpy(np.stack(labels)).float()
    t_data = torch.zeros((t_label.shape[0], maxlen)).long()
    t_weights = torch.zeros((t_label.shape[0], maxlen)).float()
    for i in range(len(data)):
        d = data[i]
        w = weights[i]
        t_data[i,:len(d)] = torch.from_numpy(d)
        t_weights[i, :len(w)] = torch.from_numpy(w)
    return t_data, t_label, t_weights

class FCN(nn.Module):
    def __init__(self, arch, sparse):
        super(FCN, self).__init__()
        self.dense = not sparse
        self.sparse = sparse

        lens = [int(i) for i in arch.split('-')]
        if self.dense:
            layers = []
            for i in range(len(lens) -1):
                layers.append(nn.Linear(lens[i], lens[i+1]))
                layers.append(nn.ReLU())
            layers = layers[:-1]
            self.model = nn.Sequential(*layers)
        else:
            self.embedding = nn.Embedding(lens[0], lens[1], padding_idx=0)
            layers = []
            for i in range(len(lens) -2):
                layers.append(nn.Linear(lens[i+1], lens[i+2]))
                layers.append(nn.ReLU())
            layers = layers[:-1]
            self.model = nn.Sequential(*layers)
            

    def forward(self, x, w=None):
        if self.dense:
            x = self.model(x)
        else:
            emb = torch.sum(self.embedding(x) * w.unsqueeze(2), axis=1).squeeze()
            x = self.model(emb)
        return x
        

class FCNRZ(nn.Module):
    def __init__(self, arch, sizes, sparse):
        super(FCNRZ, self).__init__()
        self.layers = []
        self.weights = []
        lens = [int(i) for i in arch.split('-')]
        self.dense = not sparse
        self.sparse = sparse

        if sparse:
            hashed_weight = nn.Parameter(torch.from_numpy(np.random.uniform(-1/np.sqrt(lens[0]), 1/np.sqrt(lens[0]), size=sizes[0]).astype(np.float32)))
            self.embedding = HashedEmbeddingBag(lens[0], lens[1], _weight=hashed_weight, val_offset=0, uma_chunk_size=32, no_bag=True, sparse=False, padding_idx=0)
            self.weights.append(hashed_weight)

            for i in range(len(lens) -2):
                hashed_weight = nn.Parameter(torch.from_numpy(np.random.uniform(-1/np.sqrt(lens[i]), 1/np.sqrt(lens[i]), size=sizes[i]).astype(np.float32)))
                rzlinear = RzLinear(lens[i+1], lens[i+2], 1, hashed_weight, tiled=True)
                self.weights.append(hashed_weight)
                self.layers.append(rzlinear)
                self.layers.append(nn.ReLU())
                self.layers = self.layers[:-1]
                self.model = nn.Sequential(*self.layers)
            
        else:
            for i in range(len(lens) -1):
                hashed_weight = nn.Parameter(torch.from_numpy(np.random.uniform(-1/np.sqrt(lens[i]), 1/np.sqrt(lens[i]), size=sizes[i]).astype(np.float32)))
                rzlinear = RzLinear(lens[i], lens[i+1], 1, hashed_weight, tiled=True)
                self.weights.append(hashed_weight)
                self.layers.append(rzlinear)
                self.layers.append(nn.ReLU())
                self.layers = self.layers[:-1]
                self.model = nn.Sequential(*self.layers)

    def forward(self, x, w = None):
        if self.dense:
            x = self.model(x)
        else:
            emb = torch.sum(self.embedding(x) * w.unsqueeze(2), axis=1).squeeze()
            x = self.model(emb)
        return x


def prec_compute(A, P, k):
    idx = np.argpartition(-P, axis=1, kth=k)[:,:k]
    values = [np.sum(A[i][idx[i]])/k for i in range(idx.shape[0])]
    return values
    
    
def eval(test_dataloader, model, test_itr, sparse, dev, epoch, itr, log_handle):
    #actual_labels = []
    #predicted_labels = []
    p1values = []
    p3values = []
    p5values = []

    for j,X in tqdm(enumerate(test_dataloader), total=test_itr):
        x = X[0]
        y = X[1]
        w = None
        if sparse:
            w = X[2].float().to(dev)
            x = x.long().to(dev)
        else:
            x = x.float().to(dev)
        y = np.array(y)
        #actual_labels.append(y)
        yhat = model(x, w)
        yhat = np.array(yhat.detach().cpu())
        #predicted_labels.append(yhat)
        p1values = p1values + prec_compute(y, yhat, 1)
        p3values = p3values + prec_compute(y, yhat, 3)
        p5values = p5values + prec_compute(y, yhat, 5)

    print("Metrics .. computing ..")
    #A = np.concatenate(actual_labels, axis=0)
    #P = np.concatenate(predicted_labels, axis=0)
    #acc = xc_metrics.Metrics(true_labels=A)
    #args = acc.eval(P, 5)
    #prec = str(','.join([ str(i) for i in args[0]]))
    #ndg = str(','.join([ str(i) for i in args[1]]))
    #log_handle.write("epoch," + str(epoch) + ",itr," + str(itr) + ',' + prec + ',' + ndg + '\n')

    prec = str(','.join([str(i) for i in [np.mean(p1values), np.mean(p3values), np.mean(p5values)]]))
    log_handle.write("epoch," + str(epoch) + ",itr," + str(itr) + ',' + prec + '\n')
    log_handle.flush()


def train_epoch(train_dataloader, test_dataloader, model, optimizer, eval_itr, train_itr, test_itr, epoch, sparse, dev, log_handle):
    for j,X in tqdm(enumerate(train_dataloader), total=train_itr):
        x = X[0]
        y = X[1]
        w = None
        if sparse:
            w = X[2].float().to(dev)
            x = x.long().to(dev)
            y = y.float().to(dev)
        else:
            x = x.float().to(dev)
            y = y.float().to(dev)
        

        optimizer.zero_grad()
        yhat = model(x, w)
        loss = F.binary_cross_entropy_with_logits(yhat, y,  reduction = 'mean')
        loss.backward()
        optimizer.step()
        if (epoch * train_itr + j + 1) % eval_itr == 0:
            eval(test_dataloader, model, test_itr, sparse, dev, epoch, j, log_handle)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action="store", dest="config", type=str, default=None, required=True,
                        help="config to setup the training")
    parser.add_argument('--tmpdir', action="store", dest="tmpdir", type=str, default="./runs/", help="dump dir for results")
    res = parser.parse_args()
    with open(res.config, "r") as f:
        config = yaml.load(f)
    
    temp_dir = tempfile.mkdtemp(dir=res.tmpdir)
    log_file = temp_dir + "/results.txt"
    log_handle = open(log_file, "w")

    config_name = res.config.split('/')[-1]
    with open(temp_dir + "/" + config_name, "w") as f:
        yaml.dump(config, f)

    test_dataset = GenSVMFormatParser(config['data']['test_file'], config["data"], config["sparse"])
    train_dataset = GenSVMFormatParser(config['data']['train_file'], config["data"], config["sparse"])
    if config["sparse"]:
      collate_fn = sparse_collate_function
    else:
      collate_fn = None

    train_samples = train_dataset.__len__()
    test_samples = test_dataset.__len__()
    train_itr = int(train_samples / config['data']['train_batch'])
    test_itr = int(test_samples / config['data']['test_batch'])

    train_dataloader = DataLoader(train_dataset, batch_size=config['data']['train_batch'], shuffle=True, collate_fn = collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config['data']['train_batch'], shuffle=True, collate_fn = collate_fn)
    
    if config['model']['rz']:
        model = FCNRZ(config['model']['arch'], config['model']['sizes'], config["sparse"])
    else:
        model = FCN(config['model']['arch'], config["sparse"])
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"])
    dev = config["device"]

    if dev != -1:
        model = model.float().to(dev)

    for epoch in range(config["epochs"]):
        train_epoch(train_dataloader, test_dataloader, model, optimizer,  config["eval_itr"], train_itr, test_itr, epoch, config["sparse"], dev, log_handle)
        if (epoch  + 1) % config["eval_epoch"] ==0:
            eval(test_dataloader, model, test_itr, config["sparse"], dev, epoch, 0, log_handle)

    eval(test_dataloader, model, test_itr, config["sparse"], dev, epochs, 0, log_handle)
    log_handle.close()
