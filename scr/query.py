from pathlib import Path
from typing import Optional
import numpy as np
import h5py

def calculate_NetVLAD(device, model, img):
  '''This function calculate the NetVLAD of an preprocessed image
  Args
  ------------------------------------------------------------------------------
  model: pytorch Module, with our netvlad component
  img: an image, preprocess according to pytorch format

  Return
  ------------------------------------------------------------------------------
  v: NetVLAD vector of that img
  '''
  img = img.unsqueeze(0).to(device)
  img_encoding = model.encoder(img)
  v = model.netvlad(img_encoding)
  v = v.view(-1).detach().cpu().numpy()
  return v
    
def calculate_netvlads(device, model, dataset, out_path: Optional[Path] = None):
  '''This function calculate netvlads of all image in a dataset, then store them
  inside a .h5 file
  Args
  ------------------------------------------------------------------------------
  model: pytorch model, with our netvlad component, requires two field:
    - model.encoder: image encoder
    - model.netvlad: netvlad layer
  dataset: Pytorch's Map style dataset object, require:
    - dataset.names: list of file names in database
  out_path: where we save dataset's netvlads
  '''
  if out_path is None:
    out_path = Path(__file__).absolute() / 'netvlads.h5'
  out_path.parent.mkdir(exist_ok=True, parents = True)

  for i, query in enumerate(dataset):
    name = dataset.names[i]
    v = calculate_NetVLAD(device, model, query)
    with h5py.File(str(out_path), 'a', libver='latest') as fd:
      try:
        if name in fd:
          del fd[name]
        fd[name] = v
      except OSError as error:
        if 'No space left on device' in error.args[0]:
          del grp, fd[name]
        raise error
      
def query(query_features: Path, 
          db_features: Path,
          out_path: Optional[Path]=None,
          n_result=10):
  '''This function:
    - read query's and db's netvlads
    - compare query's netvlads to db's netvlads 
    - create a retrieval dictionary
    - save that dictionary into a .h5 file 

  Args
  ------------------------------------------------------------------------------
  query_features: .h5 file containing query's netvlads
  db_features: .h5 file containing db's netvlads
  out_path: where we save our retrieval result
  '''
  def read_netvlads(path: Path):
    with h5py.File(str(path), 'r', libver='latest') as f:
      names = []
      netvlads = []
      for i, key in enumerate(f.keys()):
        names.append(key)
        netvlads.append(f[key][()])
    netvlads = np.array(netvlads)
    return names, netvlads
  
  if out_path is None:
    out_path = Path(__file__).absolute() / 'retrievals.h5'
  query_names, query_netvlads = read_netvlads(query_features)
  db_names, db_netvlads = read_netvlads(db_features)

  sim = np.einsum('id, jd -> ij', query_netvlads, db_netvlads)
  pairs = pairs_from_similarity_matrix(sim, n_result)
  pairs = [(query_names[i], db_names[j]) for i,j in pairs]
  retrieved_dict = {}

  for query_name, db_name in pairs:
    if query_name in retrieved_dict.keys():
      retrieved_dict[query_name].append(db_name)
    else:
      retrieved_dict[query_name] = [db_name]
  
  with h5py.File(str(out_path), 'w', libver = 'latest') as f:
    try:
      for k,v in retrieved_dict.items():
        f[k] = v
    except OSError as error:
      if 'No space left on device' in error.args[0]:
        pass
      raise error


def pairs_from_similarity_matrix(sim, n_results):
    """This function create pair of similar indices from a similarity matrix"""
    idx = np.argsort(sim, axis =1)
    n_col = idx.shape[1]-1
    pairs = []
    for i,_ in enumerate(sim):
        for j in range(n_results):
            pairs.append((i, idx[i,n_col-j]))
    return pairs
