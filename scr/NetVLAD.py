import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np

class NetVLADLayer(nn.Module):
    """This class implement the NetVLAD layer using pytorch.
    The ideas is mentioned in: https://arxiv.org/pdf/1511.07247.pdf
    This would be suitable to be plus into any CNNs.
    
    Args
    ------------------------------------------------------------------
    n_vocabs: number of visual words
    k: dimension of each visual word (descriptor vector)
    normalize: whether or not to normalize input embedding
    alpha: alpha as mentioned in the paper
    vocabs: the vocabulary embedding -> learnable by a NN
    conv: mapping conv layer from embedding -> netvlad vocabulary
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
        x: [N,C, w,h] - image embedding of the whole batch

        Return:
        vlad: [N, self.n_vocabs] - netvlad of the whole batch
        """
        N, C = x.size()[:2] 
        if self.normalize:
            x = F.normalize(x, p=2, dim =1)
        
        x_flatten = x.view(N,C, -1)

        soft_assignment = self.conv(x).view(N, self.n_vocabs, -1) 
        soft_assignment = F.softmax(soft_assignment, dim = 1)

        residual = x_flatten.expand(self.n_vocabs, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.vocabs.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assignment.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

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