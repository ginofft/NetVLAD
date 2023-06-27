import numpy as np
import random
import torch

class OnlineTripletSampler(torch.utils.data.BatchSampler):
  '''This class implement a sampler to used with Online Triplet Mining 
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
    self.n_batches = len(data_source) // self.P

    self.pid2imgs = self._create_pid2imgs(self.data_source)

  def __iter__(self):
    '''This function yield the batches according to BatchHardTripletMining, specifically:
      - Each batch contains P classes
      - Each class contain K images
    
    The Algorithm is as followed:
      - sample dataset into multiple base_indices - each with P images
      - Given a base_indices:
        + take an indices (index of P images)
          * create a dictionary, whose key-value are:
            key: class
            value: index of image belonging to that class
          * iterate over idx in indices -> build dictionary 
          * After which, if len(list(dict.keys()))<P (not enough classes)
            -> add new keys(of different class) until we have P classes
          * Iterate over all dict.keys() -> if len(dict[key]) < K
            -> add new random image until we have K images
        + iterate over built dictionary -> add to batch
        + if len(batch) == batch_size -> return a batch of index
        + continue (i use yield instead of return)
    '''
    batch = [] 
    base_indices = np.random.permutation(len(self.data_source))
    base_indices = np.reshape(base_indices[:(self.P*self.n_batches)],
                              (-1, self.P))
    
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
          if len(difference_set)<n_more_image:
            difference_set = np.tile(list(difference_set),
                                     n_more_image//len(difference_set)+1)
          more_image = random.sample(list(difference_set), n_more_image)
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
      return (len(self.data_source)+self.batch_size -1)//self.P
  
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
    for i, (image, label) in enumerate(data_source):
      if label not in pid2imgs:
        pid2imgs[label] = [i]
      else:
        pid2imgs[label].append(i)
    return pid2imgs