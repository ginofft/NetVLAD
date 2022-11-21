import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np
import h5py

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
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
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

def plot_retrievals_images(retrieval, query_dir:Path, db_dir: Path):
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
        query_img = [read_image(query_dir/ query_ref)]
        db_imgs = [read_image(db_dir/ r) for r in db_refs[i]]
        plot_images(query_img, dpi=25)
        plot_images(db_imgs, dpi=25)

def pairs_from_similarity_matrix(sim, n_results):
    """This function create pair of similar indices from a similarity matrix"""
    idx = np.argsort(sim, axis =1)
    n_col = idx.shape[1]-1
    pairs = []
    for i,_ in enumerate(sim):
        for j in range(n_results):
            pairs.append((i, idx[i,n_col-j]))
    return pairs
    