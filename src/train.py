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
  n_batches = len(dataloader)

  epoch_loss = 0
  start_iter = 1
  model.train()
  for batch_id, (imgs, labels) in enumerate(tqdm(dataloader), start_iter):
    #Compute netvlads embedding
    imgs, labels = imgs.to(device), labels.to(device)
    embeddings = model.encoder(imgs)
    netvlads = model.netvlad(embeddings)

    #Loss & Backprop
    loss = criterion(netvlads, labels).to(device)
    optimizer.zero_grad() #zero_grad for each batch
    loss.backward()
    optimizer.step()

    batch_loss = loss.item()
    epoch_loss += batch_loss
    
    #delete stuff to save RAM
    del imgs, labels, embeddings, netvlads
    del loss

    #Logs
    if batch_id % 50 == 0  or n_batches <= 10:
      print('Epoch[{}]({}/{}): Loss: {:.4f}'.format(epoch, batch_id,
                                                    n_batches, batch_loss),
                                                    flush=True)
    del batch_loss
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
  n_batches = len(dataloader)
  
  epoch_loss = 0
  model.eval()
  with torch.no_grad():     
    for batch_id, (imgs, labels) in enumerate(tqdm(dataloader)):
      #Compute netvlads embedding
      imgs, labels = imgs.to(device), labels.to(device)
      embeddings = model.encoder(imgs)
      netvlads = model.netvlad(embeddings)

      #Loss & Backprop
      loss = criterion(netvlads, labels).to(device)
      
      batch_loss = loss.item()
      epoch_loss += batch_loss
      
      netvlads = netvlads.cpu()
      accuracy_vector = torch.zeros(P)
      similarity_matrix = np.einsum('id, jd -> ij', netvlads, netvlads)
      sorted_indices = np.argsort(similarity_matrix, axis=1)
      for row_index,row in enumerate(sorted_indices):
        base_label = row[0]
        for index in row[1:K+1]:
          if index == base_label:
            accuracy_vector[row_index] += 1
      accuracy = torch.mean(accuracy_vector/K)

      #delete stuff to save RAM
      del imgs, labels, embeddings, netvlads
      del loss, batch_loss
      avg_loss = epoch_loss / n_batches
  print('----> Validation loss: {:.4f}'.format(avg_loss), flush=True)
  print('\n----> Retrieval Accuracy: {:.4f}'.format(accuracy*100), flush=True)

  del sampler, dataloader
  torch.cuda.empty_cache()
  return avg_loss, accuracy
