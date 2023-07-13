from torchvision import transforms
from typing import Optional
from tqdm import tqdm
import h5py
import torch
from pathlib import Path
import numpy as np

from .dataset import OnlineTripletImageDataset

class EmbeddingRetriever:
    def __init__(self, dataset: OnlineTripletImageDataset,
                path: Optional[Path] = None,
                model: Optional[torch.nn.Module] = None,
                device = torch.device("cpu")):
        self.device = device
        self.dataset = dataset
        self.model = model
        self.path = path

        self.names = []
        self.embeddings = []
        if (self.path is not None) and (self.path.suffix == '.h5'):
            print("Loading saved embedding from: ", str(self.path))
            self.names, self.embeddings = self.load_embeddings()
    
    def export_embeddings(self):
        if (self.path is None):
            self.path = Path('embeddings.h5')
        elif (self.path.suffix!='.h5'):
            self.path = self.path/'embeddings.h5'
        
        self.path.parent.mkdir(exist_ok=True, parents = True)
        if (len(self.names) == 0) or (len(self.embeddings) == 0) or (len(self.names) != len(self.embeddings)):
            self._calculate_embeddings() 
        with h5py.File(str(self.path), "a", libver="latest") as f:
            for i, name  in enumerate(self.names):
                try:
                    if name in f:
                        del f[name]
                    f[name] = self.embeddings[i]
                except OSError as error:
                    if 'No space left on device' in error.args[0]:
                        del f[name]
                    raise error
    
    def load_embeddings(self):
        embeddings = []
        with h5py.File(str(self.path), 'r', libver="latest") as r:
            names = self._get_dataset_keys(r)
            for name in names:
                embeddings.append(r[name][()])
            embeddings = np.array(embeddings)
        print("Embeddings loaded!!", flush = True)
        return names, embeddings

    def query(self, lst_image, n_results = 5):
        default_preprocessing = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225])
        ])

        query_embeddings = []
        for image in lst_image:
            image = default_preprocessing(image).unsqueeze(0)
            query_embeddings.append(self._calculate_embedding(image))
        query_embeddings = np.array(query_embeddings)

        similarity_matrix = np.einsum('id, jd -> ij', query_embeddings, self.embeddings)
        sorted_matrix = np.argsort(similarity_matrix, axis = 1)
        n_cols = sorted_matrix.shape[1]-1
        results = []
        for i, _ in enumerate(sorted_matrix):
            row = []
            for j in range(n_results):
                row.append(self.names[sorted_matrix[i, n_cols - j]])
            results.append(row)
        return results
    
    def _calculate_embedding(self, img_tensor):
        img_tensor = img_tensor.to(self.device)
        embedding = self.model.encoder(img_tensor)
        v = self.model.netvlad(embedding)
        return v.view(-1).detach().cpu().numpy()

    def _calculate_embeddings(self):
        if self.model == None:
            raise ValueError("No model found!!")

        for i, (img, label) in enumerate(tqdm(self.dataset)):
            full_path_name = str(self.dataset.root/self.dataset.names[i])
            v = self._calculate_embedding(img.unsqueeze(0))
            self.names.append(full_path_name)
            self.embeddings.append(v)
        self.embeddings = np.array(self.embeddings)
    
    def _get_dataset_keys(self, f):
        keys = []
        f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
        return keys
