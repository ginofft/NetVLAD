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
from tensorboardX import SummaryWriter

import h5py
import numpy as np
from NetVLAD import NetVLAD, NetVLADLayer

parser = argparse.ArgumentParser()

def train(epoch):
    epoch_loss = 0
    startIter = 1

    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        subsetIdx = np.array_split(np.arrange(len(train_set)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arrange(len(train_set))]
    
    nBatches = (len(train_set) + opt.batchSize -1 ) // opt.batchSize

    for subIter in range(subsetN):
        print('===> Building Cache')
        model.eval()
        train_set.cache = join(opt.cachePath, train_set.whichSet + '_feat_cache.hdf5')
        with h5py.File(train_set.cache, mode = 'w') as h5:
            pool_size = encoder_dim
            if opt.pooling.lower() == 'netvlad': pool_size *= opt.n_clusters
            h5feat = h5.create_dataset("feature",
                [len(whole_train_set), pool_size],
                dtype = np.float32)
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(whole_training_data_loader, 1):
                    
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
        
    if opt.mode.lower() == 'train':
        print("===> Training model")
        writer = SummaryWriter

        logdir = writer.file_writer.get_logdir()

        opt.savePath = join(logdir, opt.savePath)
        if not opt.resume:
            makedirs(opt.savePath)
        
        with open(join(opt.savePath, 'flags.json'), 'w') as f:
            f.write(json.dump(
                {k: v for k,v in vars(opt).items()}
            ))
        
        print('===> Saving state to: ', logdir)
        not_improved = 0
        best_score = 0

        for epoch in range(opt.start_epoch+1, opt.nEpochs+1):
            train(epoch)
            if (epoch % opt.evalEvery) == 0:
                recalls = test(whole_test_set, epoch, write_tboard = True)
                is_best = recalls[5]>best_score
                if is_best:
                    not_improved = 0
                    best_score = recalls[5]
                else:
                    not_improved += 1
                
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'recalls': recalls,
                    'best_score': best_score,
                    'optimizer': optimizer.state_dict(),
                    'parallel': isParallel,
                }, is_best)

                if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
                    print('Performance did not improve for', opt.patience, 'epochs. Stopping.')
                    break
        print('=> Best Recall@5: {:.4f}'.format(best_score), flush=True)
        writer.close()
