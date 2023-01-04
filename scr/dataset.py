import torch
from pathlib import Path
import numpy as np

from torchvision import transforms
from torch.utils.data.dataloader import default_collate

from utils import read_image

class ImageDataset(torch.utils.data.Dataset):
  '''This class is the image dataset used in inferences
    Args:
    ----------------------------------------------------------------------------
    default_conf, default_preprocessing: 
      - conf: standard stuff
      - preprocessing: resize, center crop, to Tensor (from numpy), and normalize
    self.root: Database folder, in which there are images folder 
      -> each folder would be a class
    self.names: array of Paths of image relative to root -> used to read images
  '''
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
      
      # Iterate over directories to get Images' path
      paths = []
      for g in self.default_conf['globs']:
          paths += list(Path(root).glob('**/'+g)) #read paths with file following 'globs'
      paths = sorted(list(set(paths))) 
      self.names = [i.relative_to(root).as_posix() for i in paths]
  
  def __getitem__(self, idx):
    ''' This function is due to Pytorch's Map-style dataset.
    Currently return two element:
      - query: TENSOR image of the index
      - label: label of that image  
    '''        
    query =  read_image(self.root/self.names[idx])
    if self.input_transforms:
      query = self.input_transforms(query)         
    return query
    
  def __len__(self):
      return len(self.names)

class TripletDataset(torch.utils.data.Dataset):
    '''This class load data for naive triplet loss. 
    Concretely, for an anchor image, it would load:
      - query: anchor image
      - positive: an positive image 
      - negatives: nNeg negative image
    The intuition is that, the positive image should be closer to anchor,
    compared to nNeg negative images.

    The folder structure should be:
      - database folder
          + class_1:
              * class_1_img1
              * class_1_img2
              ...
          + class_2:
            * class_2_img1
            * class_2_img2
            ...
    Args
    ----------------------------------------------------------------------------
    root: path of database folder
    nNeg: no. negative images per anchor
    input_transform: transformation done on the image
    names: path of individual images
    idx_dict: index dictionary:
      - key: class name
      - value: [index] index of images belonging to that class
    '''

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
    def __init__(self, root,
                 input_transforms = default_preprocessing,
                 nNeg=5):
        super().__init__()
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
      '''
      This method get:
        - 1 query image (anchor) 
        - 1 positive image 
        - Nneg negative images
        - indices: index of returrn images [anchor_idx, pos_idx] + [negative_indices]
      '''
      positive_dir = str(Path(self.names[idx]).parent)
      positive_indices = np.array(self.idx_dict[positive_dir])

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

#use this with dataloader for naive triplet   
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
        
class OnlineTripletImageDataset(torch.utils.data.Dataset):
  '''This class is the Image folder dataset to be used with Online Triplet Mining.
    
    Args:
    ----------------------------------------------------------------------------
    default_conf, default_preprocessing: 
      - conf: standard stuff
      - preprocessing: resize, center crop, to Tensor (from numpy), and normalize
    self.root: Database folder, in which there are images folder 
      -> each folder would be a class
    self.names: array of Paths of image relative to root -> used to read images
    self.labels: array of images' label
  '''
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
      
      # Iterate over directories to get Images' path
      paths = []
      for g in self.default_conf['globs']:
          paths += list(Path(root).glob('**/'+g)) #read paths with file following 'globs'
      paths = sorted(list(set(paths))) 
      self.names = [i.relative_to(root).as_posix() for i in paths]

      # Iterate over folder, to get classes -> then numbered them
      folders = [folder.relative_to(root).as_posix() for folder in root.iterdir()]
      temp_dict = {}
      for i, folder in enumerate(folders):
        temp_dict[folder] = i
      
      # Create labels array
      self.labels = np.zeros(len(self.names), dtype = int)
      for i, name in enumerate(self.names):
        dir = str(Path(name).parent)
        self.labels[i] = temp_dict[dir]
  
  def __getitem__(self, idx):
    ''' This function is due to Pytorch's Map-style dataset.
    Currently return two element:
      - query: TENSOR image of the index
      - label: label of that image  
    '''        
    query =  read_image(self.root/self.names[idx])
    label = self.labels[idx]       
    if self.input_transforms:
      query = self.input_transforms(query)         
    return query, label
    
  def __len__(self):
      return len(self.names)