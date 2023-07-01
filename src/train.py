import torch
from .sampler import OnlineTripletSampler
import numpy as np
from tqdm import tqdm

def train(device, 
        model, 
        epoch, 
        train_set, 
        P, K,
        criterion,
        optimizer):

  sampler = OnlineTripletSampler(P=P, K=K, data_source = train_set, drop_last = True)
  dataloader = torch.utils.data.DataLoader(
      train_set, 
      batch_sampler = sampler, 
      num_workers = 2,
      pin_memory = True
  )
  n_batches = 0 
  epoch_loss = 0
  model.train()
  for (imgs, labels) in tqdm(dataloader):
    n_batches += 1
    #Compute netvlads embedding
    imgs, labels = imgs.to(device), labels.to(device)
    embeddings = model.encoder(imgs)
    netvlads = model.netvlad(embeddings)

    #Loss & Backprop
    loss = criterion(netvlads, labels).to(device)
    optimizer.zero_grad() #zero_grad for each batch
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    
  avg_loss = epoch_loss / n_batches
  print('---> Epoch {} compledted: Train Avg. Loss: {:.4f}'.format(epoch, avg_loss),
      flush = True)
  
  del sampler, dataloader
  torch.cuda.empty_cache() #clear GPU RAM
  
  return avg_loss

def validate(device,
            model,
            val_set, 
            P, K,
            criterion):
  sampler = OnlineTripletSampler(P=P, K=K, data_source = val_set, drop_last = True)
  dataloader = torch.utils.data.DataLoader(
      val_set, 
      batch_sampler = sampler,
      num_workers = 2,
      pin_memory = True
  )
  n_batches = 0
  epoch_loss = 0
  accuracy = 0
  model.eval()
  with torch.no_grad():     
    for (imgs, labels) in tqdm(dataloader):
      n_batches += 1
      #Compute netvlads embedding
      imgs, labels = imgs.to(device), labels.to(device)
      embeddings = model.encoder(imgs)
      netvlads = model.netvlad(embeddings)

      #Loss & Backprop
      loss = criterion(netvlads, labels).to(device)
      
      batch_loss = loss.item()
      epoch_loss += batch_loss
      
      netvlads = netvlads.cpu()
      accuracy_vector = torch.zeros(P*K)
      similarity_matrix = np.einsum('id, jd -> ij', netvlads, netvlads)
      sorted_indices = np.argsort(similarity_matrix, axis=1)
      for row_index,row in enumerate(sorted_indices):
        row = row[::-1]
        base_label = labels[row[0]]
        for index in row[1:K+1]:
          if labels[index] == base_label:
            accuracy_vector[row_index] += 1
      accuracy += torch.mean(accuracy_vector/K)
    
  avg_loss = epoch_loss / n_batches
  accuracy = 100* (accuracy / n_batches)
  print('----> Validation Loss/Accuracy: {:.4f} / {:.4f}'.format(avg_loss, accuracy), flush=True)

  del sampler, dataloader
  torch.cuda.empty_cache()
  return avg_loss, accuracy
