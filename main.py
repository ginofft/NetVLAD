import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import VGG16_Weights

from scr.netvlad import NetVLADLayer
from scr.dataset import OnlineTripletImageDataset, ImageDataset
from scr.loss import OnlineTripletLoss
from scr.utils import save_checkpoint, load_checkpoint, plot_retrievals_images, str2bool
from scr.train import train, validate
from scr.query import query, calculate_netvlads

parser = argparse.ArgumentParser(description = 'torch-netvlad-online_triplet_mining')
#Hyper Parameters
##Sampler 
parser.add_argument('--P', type=int, default=4, 
                    help='no. classes for Online Triplet Mining')
parser.add_argument('--K', type=int, default=8, 
                    help='no. images per class for Online Triplet Mining')
##Optimizer
parser.add_argument('--optim', type=str, default = 'Adam', 
                    help='optimizer to use', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
### SGD Scheduler params
parser.add_argument('--lrStep', type=int, default = 5, 
                    help='no. step before LR decay')
parser.add_argument('--lrGamma', type=float, default=0.5, 
                    help='Decay constant')
##Loss
parser.add_argument('--margin', type=float, default=0.1**0.5, 
                    help='Margin for triplet loss')
parser.add_argument('--tripletLoss', type=str, default='batchall', 
                    help='Type of triplet loss to use. There are three available: naive (random triplet), online triplet mining - hard variation, online triplet mining - semi-hard variation',
                    choices=['naive', 'batchall', 'batchhard'])
##NetVLAD
parser.add_argument('--n_vocabs', type=int, default=16, 
                    help='no. netvlad vocabulary')

#Training Arguments 
parser.add_argument('--nEpochs', type =  int, default = 500, help='no. epochs')
parser.add_argument('--mode', type=str, default='train', 
                    help='Traning mode or Testing(inference) mode', 
                    choices=['train', 'test'],
                    required=True)
parser.add_argument('--trainPath', type=str, default='', 
                    help='Path of training set',
                  )
parser.add_argument('--validationPath', type=str, default='', 
                    help='Path of validation set')
parser.add_argument('--savePath', type=str, default='', 
                    help='Path to save checkpoint to')
parser.add_argument('--loadPath', type=str, default='', 
                    help='Path to load checkpoint from - used for resume or testing')
parser.add_argument('--oldLoss', type = str2bool, nargs = '?', default= True,
                    help='If true, resume training with stored val and train loss; Else, set val and train loss equal 1.\n You should set this to false when switching loss function'
                    )
parser.add_argument('--saveEvery', type=int, default=25, 
                    help='no. epoch before a save is created')

#Inference Arguments
parser.add_argument('--dbPath', type=str, default='', 
                    help='Path to database folder (NOT training set folder)')
parser.add_argument('--queryPath', type=str, default='',
                    help='Path to query folder')
parser.add_argument('--outPath', type=str, default='', 
                    help="Path where to store: database's netvlads, query's netvlads and retrieval results")
# parser.add_argument('--plotRetrieval', type=bool, default=False,
#                     help='Whether or not to plot out retrieval results (currently have some problem with notebook)')

if __name__ == "__main__":
  opt = parser.parse_args()
  cuda = torch.cuda.is_available()
  if cuda:
    device =  torch.device("cuda")
  else:
    raise Exception("No GPU found, please get one")
  #Setup model
  if opt.loadPath:
    encoder = models.vgg16(weights=None)
  else:
    encoder = models.vgg16(weights=VGG16_Weights.DEFAULT)

  encoder_k = 512 ##TODO
  layers = list(encoder.features.children())[:-2]

  model = nn.Module()
  encoder = nn.Sequential(*layers)
  model.add_module('encoder', encoder)

  net_vlad = NetVLADLayer(n_vocabs = opt.n_vocabs, k = encoder_k)
  model.add_module('netvlad', net_vlad)

  model = model.to(device)

  if opt.mode.lower() == 'train':
    startEpoch = 0
    val_loss = 1
    train_loss = 1

    if opt.trainPath:
      train_set = OnlineTripletImageDataset(Path(opt.trainPath))
    else:
      raise Exception("Please provide a trainset using --trainPath")
    if opt.validationPath:
      val_set = OnlineTripletImageDataset(Path(opt.validationPath))
    else:
      print('No validation set found, you can set a validation set using --validationPath')

    if opt.tripletLoss.lower() == 'batchhard':
      criterion = OnlineTripletLoss(margin = opt.margin, hard=True).to(device)
    elif opt.tripletLoss.lower() == 'batchall':
      criterion = OnlineTripletLoss(margin = opt.margin, hard=False).to(device)
    elif opt.tripletLoss.lower() == 'naive':
      raise Exception('naive triplet is not implemented yet\n(cause im lazy, deal with it)')
    if opt.optim.lower() =='adam':
      optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr = opt.lr)
    else:
      pass #TODO SGD optimizer

    if opt.loadPath: #loading stuff
      startEpoch, train_loss, val_loss = load_checkpoint(Path(opt.loadPath),
                                                        device,
                                                        model, 
                                                        optimizer)
    if opt.oldLoss == False: #condition for when you switch loss function
      train_loss = 1
      val_loss = 1  
    
    
    for epoch in range(startEpoch+1, opt.nEpochs+1):
      # train & validate
      epoch_train_loss = train(device, model, epoch,
                            train_set, opt.P, opt.K,
                            criterion, optimizer)
      if opt.validationPath:
        epoch_val_loss = validate(device, model, 
                                  val_set, opt.P, opt.K,
                                  criterion)
      else:
        epoch_val_loss = 1
      #saving stuff
      if (epoch_train_loss < train_loss): #lowest loss on train set
        train_loss = epoch_train_loss
        save_checkpoint({
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, Path(opt.savePath), 'best_train.pth.tar')

      if (epoch_val_loss < val_loss): #lowest loss on val set
        val_loss = epoch_val_loss
        save_checkpoint({
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, Path(opt.savePath), 'best.pth.tar')
        
      if (epoch % opt.saveEvery) == 0: #save every epoch
        save_checkpoint({
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, Path(opt.savePath), 'epoch{}.pth.tar'.format(epoch))      
  else:  
    if opt.loadPath: #loading stuff
      startEpoch, train_loss, val_loss = load_checkpoint(Path(opt.loadPath), 
                                                        device,
                                                        model)
    else:
      raise Exception('Please point to a model using --loadPath')

    if not opt.dbPath: 
      raise Exception('Please provide database folder using --dbPath')
    if not opt.queryPath:
      raise Exception('Please provide query folder using --queryPath')
    #Set up output paths
    query_features = Path(opt.outPath) / 'q_features.h5'
    db_features = Path(opt.outPath) / 'db_features.h5'
    retrieval = Path(opt.outPath) / 'retrieved.h5'
    
    #Load database into Dataset, then calculate db's netvlads
    db_dataset = ImageDataset(Path(opt.dbPath))
    calculate_netvlads(device, model, db_dataset, db_features)
    #Load query into Dataset, then calculate query's netvlads
    query_dataset = ImageDataset(Path(opt.queryPath))
    calculate_netvlads(device, model, query_dataset, query_features)

    #Find Retrieval 
    query(query_features, db_features, retrieval)
    #plot_retrievals_images(retrieval, Path(opt.dbPath), Path(opt.queryPath))
