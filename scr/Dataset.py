import torch
from pathlib import Path
import numpy as np
import collections  

from .utils import read_image

class TripletDataset(torch.utils.data.Dataset):
    default_conf = {
      'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
      'grayscale': False,
      'interpolation': 'cv2_area'
    }
    def __init__(self, root, nNegSample=500,  nNeg=5):
        super().__init__()
        self.nNegSample = nNegSample
        self.nNeg = nNeg
        self.root = root
        paths = []

        for g in self.default_conf['globs']:
            paths += list(Path(root).glob('**/'+g)) #read paths with file following 'globs'
        paths = sorted(list(set(paths))) 
        self.names = [i.relative_to(root).as_posix() for i in paths]

        self.idx_dict = collections.defaultdict(list)
        for i,name in enumerate(self.names):
            dir = str(Path(name).parent)
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
        
        query =  [read_image(self.root/self.names[idx])]
        positive = [read_image(self.root/self.names[p_idx])]
        negatives = []
        for indices in n_indices:
            negatives.append(read_image(self.root/self.names[indices]))

        return query, positive, negatives, [idx, p_idx]+n_indices.tolist()
    
    def __len__(self):
        return len(self.names)

