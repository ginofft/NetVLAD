import torch
import torch.nn.functional as F

class OnlineTripletLoss(torch.nn.Module):
  def __init__(self, margin = 0.1**0.5, hard=True, squared=False):
    super(OnlineTripletLoss, self).__init__()
    self.margin = margin
    self.hard = hard
    self.squared = squared
  
  def forward(self, embeddings, labels):
    '''
    This function calculate the Triplet Loss using Online Triplet Mining.
    
    Args
    ------------------------------------------------------------------------------------------
    embeddings: [batch_size, d] embeddings of a batch - where d is dimension of each embedding
    labels: [batch_size] labels of a batch

    Return triplet_loss: a float number of the loss from Online Triplet Mining

    Algorithms
    ------------------------------------------------------------------------------------------
    There are two kind of triplet loss associated with Online Triplet Mining: batch hard and batch all

    Batch Hard: 
      - Take the hardest positive pair, and the hardest negative pair in the batch.
      - Calculate Triplet Loss from those two pairs.

    Batch All:
      - Take all posible triplet pairs
      - Take all 'hard' pairs - where (positive distance - negative distance + margin)>0
      - Calculate triplet loss for these 'hard' pairs
      - Average out the loss -> Final triplet Loss
    
    DO NOT use batch hard at the early stages of training, as it lead to model collapse
    Recommended workflow: train using Batch All till convergence -> switch to Batch Hard for 1-3 epochs
    '''
    pairwise_dist = self._pairwise_distance(embeddings, squared=self.squared)
    
    if self.hard: #Batch Hard
      # Get hardest positive pairs
      mask_a_pos = self._get_anchor_pos_triplet_mask(labels).float()
      valid_p_dist = pairwise_dist * mask_a_pos
      hardest_pos_dist, _ = torch.max(valid_p_dist, dim =1, keepdim = True)
      
      # Get the hardest negative pairs
      mask_a_neg = self._get_anchor_neg_triplet_mask(labels).float()
      max_a_neg_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)

      valid_neg_dist = pairwise_dist + max_a_neg_dist * (1.0 - mask_a_neg)
      hardest_neg_dist, _ = torch.min(valid_neg_dist, dim=1, keepdim=True)
      
      #Calculate BatchHard Triplet Loss
      triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
      triplet_loss = torch.mean(triplet_loss)
    else: # Batch ALL
      anc_p_dist = pairwise_dist.unsqueeze(dim=2) #[batch_size, batch_size, 1]
      anc_neg_dist = pairwise_dist.unsqueeze(dim=1) #[batch_size, 1, batch_size]
      loss = anc_p_dist - anc_neg_dist + self.margin

      mask = self._get_triplet_mask(labels).float()

      triplet_loss = loss*mask
      triplet_loss = F.relu(triplet_loss)

      hard_triplets = torch.gt(triplet_loss, 1e-16).float()
      num_hard_triplets = torch.sum(hard_triplets)
      triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

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

  def _get_triplet_mask(self, labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
      A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Check that i, j and k are distinct
    indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j * (i_equal_k ^ 1)

    mask = distinct_indices * valid_labels   # Combine the two masks

    return mask