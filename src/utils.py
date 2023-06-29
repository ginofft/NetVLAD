import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image
import h5py
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models
from .netvlad import NetVLADLayer
import argparse

ENCODER_MAPPING = {
    'VGG13': {'model': models.vgg13, 'pretrained_weight' : models.VGG13_Weights.IMAGENET1K_V1, 'output_channel_size' : 512},
    'VGG16': {'model': models.vgg16, 'pretrained_weight' : models.VGG16_Weights.IMAGENET1K_V1, 'output_channel_size' : 512},
    'VGG19': {'model': models.vgg19, 'pretrained_weight' : models.VGG19_Weights.IMAGENET1K_V1, 'output_channel_size' : 512},
    'Resnet18': {'model': models.resnet18, 'pretrained_weight' : models.ResNet18_Weights.IMAGENET1K_V1, 'output_channel_size' : 512},
    'Resnet34': {'model': models.resnet34, 'pretrained_weight' : models.ResNet34_Weights.IMAGENET1K_V1, 'output_channel_size' : 512},
    'Resnet50': {'model': models.resnet50, 'pretrained_weight' : models.ResNet50_Weights.IMAGENET1K_V1, 'output_channel_size' : 2048},
    'Resnet101': {'model': models.resnet101, 'pretrained_weight' : models.ResNet101_Weights.IMAGENET1K_V1, 'output_channel_size' : 2048},
    'Resnet152': {'model': models.resnet152, 'pretrained_weight' : models.ResNet152_Weights.IMAGENET1K_V1, 'output_channel_size' : 2048},
}

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

    image = Image.open(path).convert('RGB')
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    return image

def plot_retrieval_images(retrieval,db_dir: Path, query_dir:Optional[Path] = None):
    """This function plots queries and retrieved images
    Args
    ----------------------------------------------------------------
    retrieval: path of .h5 file storing retrievel results
    query_dir: path of folder containing queries images
    db_dir: path of folder containing database images
    """
    with h5py.File(str(retrieval), 'r', libver='latest')as f:
        query_refs = list(f.keys())
        db_refs = []
        for key in f.keys():
            data = f[key][()]
            data = [x.decode() for x in data]
            db_refs.append(data)

    for i, query_ref in enumerate(query_refs):
        if query_dir is not None:
            query_img = [read_image(query_dir/ query_ref)]
            plot_images(query_img, dpi=25)
        db_imgs = [read_image(db_dir/ r) for r in db_refs[i]]
        plot_images(db_imgs, dpi=25)
        
def save_checkpoint(state, path:Path, filename='lastest.pth.tar'):
  out_path = path / filename
  torch.save(state, out_path)

def load_checkpoint(path, device, model, optimizer = None):
  state = torch.load(path)
  epoch = state['epoch']
  train_loss = state['train_loss']
  val_loss = state['val_loss']

  model.load_state_dict(state['model'])
  model = model.to(device)
  if optimizer != None:
    optimizer.load_state_dict(state['optimizer'])
  print("=> loaded checkpoint '{}' (epoch {})".format(True, epoch))
  print("Checkpoint's train loss is: {:.4f}".format(train_loss))
  print("Checkpoint's validation loss is: {:.4f}".format(val_loss))
  return epoch, train_loss, val_loss
  
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')
    
def construct_model(architect: str, n_vocabs: int, use_pretrained_weight = True) -> torch.nn.Module:
    architect_dict = ENCODER_MAPPING[architect]
    if use_pretrained_weight:
        encoder = architect_dict['model'](weights = architect_dict['pretrained_weight'])
    else:
        encoder = architect_dict['model'](weights = None)

    layers = list(encoder.children())[:-2]
    encoder = nn.Sequential(*layers)

    encoder_k = architect_dict['output_channel_size']
    net_vlad = NetVLADLayer(n_vocabs = n_vocabs, k = encoder_k)
    model = nn.Module()
    model.add_module('encoder', encoder)
    model.add_module('netvlad', net_vlad)
    return model