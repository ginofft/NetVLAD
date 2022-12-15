import torch
from pathlib import Path
import numpy as np
import collections  
import random

from torchvision import transforms
from torch.utils.data.dataloader import default_collate

from .utils import read_image

#naive Triplet - Its fucking suck
class TripletDataset(torch.utils.data.Dataset):
    default_conf = {
      'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
      'grayscale': False,
      'interpolation': 'cv2_area'
    }
    default_preprocessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225]) 
    ])
    def __init__(self, root, input_transforms = default_preprocessing, nNegSample=500,  nNeg=5):
        super().__init__()
        self.nNegSample = nNegSample
        self.nNeg = nNeg
        self.root = root
        self.input_transforms = input_transforms

        paths = []

        for g in self.default_conf['globs']:
            paths += list(Path(root).glob('**/'+g)) #read paths with file following 'globs'
        paths = sorted(list(set(paths))) 
        self.names = [i.relative_to(root).as_posix() for i in paths]

        self.idx_dict = {}
        for i,name in enumerate(self.names):
            dir = str(Path(name).parent)
            if dir not in self.idx_dict:
              self.idx_dict[dir] = [i]
            else:
              self.idx_dict[dir].append(i)

    def __getitem__(self, idx):
        #TODO use cache for faster getitem ??
        #Get positive indices
        positive_dir = str(Path(self.names[idx]).parent)
        positive_indices = np.array(self.idx_dict[positive_dir])
        #TODO - better way to get negative indices 
        #Get negative indices - 
        negative_indices = []

        keys = set(self.idx_dict.keys()).difference(set([positive_dir]))
        for key in keys:
            negative_indices.extend(self.idx_dict[key])
        negative_indices = np.array(negative_indices)
        
        #get query, positive and negative images
        p_idx = np.random.choice(positive_indices, 1).item()
        n_indices = np.random.choice(negative_indices, self.nNeg)
        
        query =  read_image(self.root/self.names[idx])
        positive = read_image(self.root/self.names[p_idx])
        
        negatives = []
        for indices in n_indices:
            negatives.append(read_image(self.root/self.names[indices]))

        if self.input_transforms:
          query = self.input_transforms(query)
          positive = self.input_transforms(positive)
          for i, neg in enumerate(negatives):
            negatives[i] = self.input_transforms(neg)

        negatives = torch.stack(negatives, 0)
        return query, positive, negatives, [idx, p_idx]+n_indices.tolist()
    
    def __len__(self):
        return len(self.names)
        
#Image dataset used with BatchHardTripletSampler
class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
      'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
      'grayscale': False,
      'interpolation': 'cv2_area'
    }
    default_preprocessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225]) 
    ])
    def __init__(self, root, input_transforms = default_preprocessing):
        super().__init__()
        self.root = root
        self.input_transforms = input_transforms
        paths = []
        for g in self.default_conf['globs']:
            paths += list(Path(root).glob('**/'+g)) #read paths with file following 'globs'
        paths = sorted(list(set(paths))) 
        self.names = [i.relative_to(root).as_posix() for i in paths]

        folders = [folder.relative_to(root).as_posix() for folder in root.iterdir()]
        temp_dict = {}
        for i, folder in enumerate(folders):
          temp_dict[folder] = i
        
        self.idx_dict = {}
        for i, name in enumerate(self.names):
          dir = str(Path(name).parent)
          if temp_dict[dir] not in self.idx_dict:
            self.idx_dict[temp_dict[dir]] = [i]
          else:
            self.idx_dict[temp_dict[dir]].append(i)

    def __getitem__(self, idx):        
        query =  read_image(self.root/self.names[idx])
        label = str(Path(self.names[idx]).parent)       
        if self.input_transforms:
          query = self.input_transforms(query)         
        return query, label
      
    def __len__(self):
        return len(self.names)

class BatchHardTripletSampler(torch.utils.data.BatchSampler):
  '''This class implement a sampler to used with BatchHard Triplet Mining 
  strategy as described in: https://arxiv.org/pdf/1703.07737.pdf

  This create batches, each containing with P*K images, where:
    - P class would be chosen
    - from each class, K image is chosen
  We divide the dataset into len(dataset)//P group - each with P images
  -> Then randomize the remaining (P*K - P) images

  Args
  ------------------------------------------------------------------------------
  self.P: no. classes per batch
  self.K: no. images per class
  self.batch_size: P*K
  self.data_source: dataset object
  self.drop_last: whether or not to drop the last unfinished batch

  self.pid2imgs: a dictionary, where:
    - keys: class label (numbered).
    - values: index of images belong to a specific class.
  '''
  def __init__(self, P, K, data_source, drop_last = True):
    self.P = P
    self.K = K
    self.batch_size  =  self.P * self.K
    self.data_source = data_source
    self.drop_last = drop_last

    self.pid2imgs = self._create_pid2imgs(self.data_source)

  def __iter__(self):
    '''This function yield the batches according to BatchHardTripletMining, specifically:
      - Each batch contains P classes
      - Each class contain K images
    
    The Algorithm is as followed:
      - sample dataset into multiple base_indices - each with P images
      - 
    '''
    batch = [] 
    
    base_indices = np.random.permutation(len(self.data_source))
    base_indices = np.reshape(base_indices, (-1, self.P))

    for indices in base_indices:
      batch_dict = {}
      
      for index in indices:
        if self.data_source.labels[index] not in batch_dict:
          batch_dict[self.data_source.labels[index]] = [index]
        else:
          batch_dict[self.data_source.labels[index]].append(index)
      
      if len(batch_dict)<self.P:
        difference_set = set(self.pid2imgs.keys()).difference(set(batch_dict.keys()))
        n_more_class = self.P - len(batch_dict)
        more_class = random.sample(difference_set, n_more_class)
        for cls in more_class:
          batch_dict[cls] = []
      
      for cls in batch_dict.keys():
        if len(batch_dict[cls]) < self.K:
          difference_set = set(self.pid2imgs[cls]).difference(set(batch_dict[cls]))
          n_more_image = self.K - len(batch_dict[cls])
          more_image = random.sample(difference_set, n_more_image)
          for image in more_image:
            batch_dict[cls].append(image)
      
      for cls in batch_dict.keys():
        for i in batch_dict[cls]:
          batch.append(i)
      
      if len(batch) == self.batch_size:
        yield batch
        batch = []
      if len(batch) > 1 and not self.drop_last:
        yield batch

  def __len__(self):
    if self.drop_last:
      return len(self.data_source)//self.P
    else:
      return (len(self. data_source)+self.batch_size -1)//self.P
  
  def _create_pid2imgs(self, data_source):
    ''' Create a label mapping between index and label from a dataset
    Args
    ---------------------------------------------------------------------
    data_source: a Pytorch dataset with a label array - whose element are integer

    Return:
    ---------------------------------------------------------------------------
    pid2imgs: a label dictionary; Whose keys are labels, and values are index belonging to that label
    '''
    pid2imgs = {}
    for i, label in enumerate(data_source.labels):
      if label not in pid2imgs:
        pid2imgs[label] = [i]
      else:
        pid2imgs[label].append(i)
    
    return pid2imgs
      

def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).
    
    Args:
        data: list of tuple (query, positive, negatives). 
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = default_collate(query)
    positive = default_collate(positive)
    negCounts = default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices
    