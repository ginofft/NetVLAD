import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np

class NetVLADLayer(nn.Module):
    """This class implement the NetVLAD layer using pytorch.
    This would be suitable to be plus into any CNNs.
    
    Args
    ------------------------------------------------------------------
    n_vocabs: number of visual words
    k: dimension of each visual word (descriptor vector)

    """
    def __init__(self, n_vocabs, k, normalize = True):
        super(NetVLADLayer, self).__init__()
        self.n_vocabs = n_vocabs
        self.k = k
        self.normalize = normalize
        self.alpha = 0
        self.conv = nn.Conv2d(self.k, self.n_vocabs, kernel_size=(1,1), bias=False)

        self.vocabs = nn.Parameter(torch.rand(self.n_vocabs, self.k))
    
    def _init_param(self, descriptors, vocabs):
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(descriptors)
        del descriptors #save RAM
        dsSq = np.square(knn.kneighbors(vocabs, 2)[0])


    def forward(self, x):
        """This function forward output from a CNNs into NetVLAD layer
        Args
        ----------------------------------------------------------
        x: [N,C, w,h] - image embedding of the whole dataset
        """
        N, C = x.size()[:2] 
        if self.normalize:
            x = F.normalize(x, p=2, dim =1)
        
        x_flatten = x.view(N,C, -1)

        soft_assignment = self.conv(x).view(N, self.n_vocabs, -1) 
        soft_assignment = F.softmax(soft_assignment, dim = 1)

        vlad = torch.zeros([N, self.n_vocabs, c],
                        dtype = x.dtype,
                        layout = x.layout
                        device = x.device)
        
        for i in range(self.n_vocabs):
            vocab_tensor = self.vocabs[i:i+1]\
                .expand(x_flatten.size()(-1), -1, -1)\
                .permute(1,2,0).unsqueeze(0)
            res = x_flatten