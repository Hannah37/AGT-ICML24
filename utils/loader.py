import torch
import numpy as np
import pandas as pd
import sys
import os

from torch.linalg import eigh
from os import walk
from .utility import *
from torch.utils.data import TensorDataset, DataLoader

def load_saved_data(args):
    adjacencies = torch.load(os.path.join(args.data_path, 'A.pt'))
    features = torch.load(os.path.join(args.data_path, 'X.pt'))
    labels = torch.load(os.path.join(args.data_path, 'labels.pt'))
    eigenvalues = torch.load(os.path.join(args.data_path, 'eigval.pt'))
    eigenvectors = torch.load(os.path.join(args.data_path, 'eigvec.pt'))

    return adjacencies, features, labels, eigenvalues, eigenvectors 


def build_data_loader(args, idx_pair, adjacencies, features, labels, eigenvalues, eigenvectors):
    idx_train, idx_test = idx_pair

    if args.model == 'svm' or args.model == 'mlp' or args.model == 'gcn' or args.model == 'gat' or args.model == 'gdc':  
        data_train = TensorDataset(adjacencies[idx_train], features[idx_train], labels[idx_train])
        data_test = TensorDataset(adjacencies[idx_test], features[idx_test], labels[idx_test])
    elif args.model == 'graphheat' or args.model == 'exact' or args.model == 'agt':
        data_train = TensorDataset(adjacencies[idx_train], features[idx_train], labels[idx_train], eigenvalues[idx_train], eigenvectors[idx_train])
        data_test = TensorDataset(adjacencies[idx_test], features[idx_test], labels[idx_test], eigenvalues[idx_test], eigenvectors[idx_test])

    data_loader_train = DataLoader(data_train, batch_size=idx_train.shape[0], shuffle=True) # Full-batch
    data_loader_test = DataLoader(data_test, batch_size=idx_test.shape[0], shuffle=True) # Full-batch
    
    return data_loader_train, data_loader_test
