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
  subsetN = 1
  subsetIdx = [np.arange(len(dataset))]

  nBatches = (len(dataset) + batchSize -1) // batchSize

  for subIter in range(subsetN):
    model.eval()
    model.train()

    sub_train_set = Subset(dataset=dataset, indices = subsetIdx[subIter])
    dataloader = DataLoader(dataset=sub_train_set, num_workers = 0, 
                            batch_size = batchSize, shuffle = True,
                            collate_fn = collate_fn)
    for iteration, (query, positive, negative, negCounts, indices) in \
      enumerate(dataloader, startIter):

      B, C, H, W = query.shape
      nNeg = torch.sum(negCounts)
      input = torch.cat([query, positive, negative])

      input = input.to(device)
      image_encoding = model.encoder(input)
      vlad_encoding = model.vlad(image_encoding)

      vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])

      optimizer.zero_grad()

      loss = 0
      for i, negCount in enumerate(negCounts):
        for n in range(negCount):
          negIx = (torch.sum(negCounts[:i]) + n).item()
          loss += criterion(vladQ[i:i+1], vladP[i:i+1], vladN[negIx:negIx+1])

      loss /= nNeg.float().to(device)

      loss.backward()
      optimizer.step()

      del input, image_encoding, vlad_encoding, vladQ, vladP, vladN
      del query, positive, negative

      batch_loss = loss.item()
      epoch_loss += batch_loss

      if iteration % 50 == 0 or nBatches <= 10:
        print("Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration,
                                                           nBatches, batch_loss), flush=True)
        print('Allocated: ', torch.cuda.memory_allocated())
        print('Cached: ', torch.cuda.memory_reserved())

    startIter += len(dataloader)
    del dataloader, loss
    optimizer.zero_grad()
    torch.cuda.empty_cache()
  avg_loss = epoch_loss / nBatches
  print("---> Epoch {} complete: Avg. Loss: {:.4f}".format(epoch, avg_loss),
        flush=True)



if __name__ == '__main__':
    opt = parser.parse_args()
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
