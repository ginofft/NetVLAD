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
    def __init__(self, n_vocabs, k, alpha=100.0, normalize = True):
        super(NetVLADLayer, self).__init__()
        self.n_vocabs = n_vocabs
        self.k = k
        self.normalize = normalize
        self.alpha = alpha
        self.conv = nn.Conv2d(self.k, self.n_vocabs, kernel_size=(1,1), bias=False)
        self.vocabs = nn.Parameter(torch.rand(self.n_vocabs, self.k))
        self._init_params()
    
    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.vocabs).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.vocabs.norm(dim=1)
        )
      

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

        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assignment.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

class NetVLAD(nn.Module):
    def __init__(self, encoder, net_vlad):
        super(NetVLAD, self).__init__()
        self.encoder = encoder
        self.NetVLAD = net_vlad

    def forward(self, x):
        x = self.encoder(x)
        embedded_x = self.NetVLAD(x)
        return embedded_x

