from pathlib import Path
from typing import Optional
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import VGG16_Weights

from PIL import Image
from typing import Optional
import argparse

def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=.5,
                adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.size[0] / i.size[1] for i in imgs]  # W / H
    else:
        ratios = [4/3] * n
    figsize = [sum(ratios)*4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios})
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)
    

def read_image(path: Path):
    """This function read an image from a path.
    The read is perform using PIL.Image (cause PyTorch).
    """

    image = Image.open(path)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    return image

    
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


def pairs_from_similarity_matrix(sim, n_results):
    """This function create pair of similar indices from a similarity matrix"""
    idx = np.argsort(sim, axis =1)
    n_col = idx.shape[1]-1
    pairs = []
    for i,_ in enumerate(sim):
        for j in range(n_results):
            pairs.append((i, idx[i,n_col-j]))
    return pairs


def query_one(image, device, model,
          db_features: Path,
          n_result=10):
  '''This function:
    - read image's and db's netvlads
    - compare image's netvlads to db's netvlads 
    - create a retrieval dictionary
    - save that dictionary into a .h5 file 
  Args
  ------------------------------------------------------------------------------
  image_vector: query_image's NetVLAD vector
  db_features: .h5 file containing db's netvlads
  out_path: where we save our retrieval result
  '''
  transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(256),
      transforms.ToTensor(),
      transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225]) 
  ])
  image = transform(image)
    
  def read_netvlads(path: Path):
    with h5py.File(str(path), 'r', libver='latest') as f:
      names = []
      netvlads = []
      for i, key in enumerate(f.keys()):
        names.append(key)
        netvlads.append(f[key][()])
    netvlads = np.array(netvlads)
    return names, netvlads
  
  image_vector = calculate_NetVLAD(device, model, image)
  query_names, query_netvlads = "result_one", image_vector
  db_names, db_netvlads = read_netvlads(db_features)

  query_netvlads = torch.from_numpy(query_netvlads)
  query_netvlads = torch.unsqueeze(query_netvlads, 0)
  sim = np.einsum('id, jd -> ij', query_netvlads, db_netvlads)
  pairs = pairs_from_similarity_matrix(sim, n_result)
  pairs = [(query_names, db_names[j]) for i,j in pairs]
  retrieved_dict = {}

  for query_name, db_name in pairs:
    if query_name in retrieved_dict.keys():
      retrieved_dict[query_name].append(db_name)
    else:
      retrieved_dict[query_name] = [db_name]

  return retrieved_dict["result_one"]


def plot_retrieval_images_one(query_img, retrieved_dict, db_dir: Path):
    """This function plots queries and retrieved images
    Args
    ----------------------------------------------------------------
    retrieved_dict: list storing retrievel results
    db_dir: path of folder containing database images
    """
    db_refs = [db_index for db_index in retrieved_dict]
#     plot_images([query_img], dpi = 25)
    plt.imshow(query_img)
    db_imgs = [read_image(db_dir+ "/"+r) for r in db_refs]
    plot_images(db_imgs[:10], dpi = 25)