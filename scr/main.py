import argparse
from math import log10, ceil
import random, shutil, json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.models as models

import h5py
import numpy as np
from NetVLAD import NetVLAD, NetVLADLayer

parser = argparse.ArgumentParser()

if __name__ == '__main__':
    opt = parser
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")
    print("====> Loadiing dataset")
    if opt.mode.lower() == 'train':
        #TODO - load training dataset
        pass
    print("======> Building Model")

    encoder_dim = 512
    encoder = models.vgg16()
    layers = list(encoder.features.children())[:-2]

    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module('encoder', encoder)

    netVLAD = NetVLADLayer(16, encoder_dim)
    model.add_module('VLAD', netVLAD)

    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() >1:
        model.encoder = nn.DataParallel(model.encoder)
        isParallel = True
    
    model = model.to(device)
    if opt.mode.lower() == 'train':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
            model.parameters()), lr = opt.lr)
    
        criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, p=2,
            reduction='sum').to(device)
        

