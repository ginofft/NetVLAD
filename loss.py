import torch
import torch.nn.functional as F

class BatchHardTripletLoss(nn.Module):
  def __init__(self, margin = 0.1**0.5, hard=True, squared=False):
    super(BatchHardTripletLoss, self).__init__()
    self.margin = margin
    self.hard = hard
    self.squared = squared
  
  def forward(self, embeddings, labels):
    '''
    This function return BatchHardTripletLoss of a batch. The process is:
      - Calculate pairwise distance - distances between embedding
      - Create two masks based on labels:
        + postive mask: same labels
        + negative mask: different labels
      - Tak
    '''

    pairwise_dist = self._pairwise_distance(embeddings, squared=self.squared)
    
    if self.hard: #Batch Hard
      # Get hardest positive pairs
      mask_a_pos = self._get_anchor_pos_triplet_mask(labels).float()
      valid_p_dist = pairwise_dist * mask_a_pos
      hardest_pos_dist, _ = torch.max(valid_p_dist, dim =1, keepdim = True)
      
      # Get the hardest negative pairs
      mask_a_neg = self._get_anchor_neg_triplet_mask(labels).float()
      max_a_neg_dist, _  = torch.max(pairwise_dist, dim=1, keepdim=True)
      a_neg_dist = pairwise_dist + max_a_neg_dist * (1.0 - mask_a_neg)
      hardest_neg_dist, _ = torch.min(a_neg_dist, dim=1, keepdim=True)
      
      #Calculate Triplet Loss
      triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
      triplet_loss = torch.mean(triplet_loss)
    else: #Batch All
      pass
    
    return triplet_loss

  def _pairwise_distance(self, x, squared=False, eps =1e-16):
    cor_mat = torch.matmul(x, x.t())
    norm_mat = cor_mat.diag()
    dist = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    dist = F.relu(dist)

    if not squared:
      mask = torch.eq(dist, 0.0).float()
      dist = dist + mask*eps
      dist = torch.sqrt(dist)
      dist = dist * (1.0-mask)

    return dist
  
  def _get_anchor_pos_triplet_mask(self, labels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1

    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask =  indices_not_equal * labels_equal

    return mask
  
  def _get_anchor_neg_triplet_mask(self, labels):
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ 1
    return mask