import os
from PIL import Image
import numpy as np
import  pandas as pd
from fastapi import FastAPI, File, UploadFile
import uvicorn
from typing import List
from pathlib import Path
from tqdm import tqdm
import base64
import cv2
from scr.query_one import query_one, plot_retrieval_images_one
from scr.utils import read_image
from scr.netvlad import NetVLADLayer

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import VGG16_Weights

app = FastAPI()

savePath = './model/BatchAll'
loadPath = 'model/BatchAll/best.pth.tar'
outPath = 'out'
db_features = Path(outPath) / 'db_features.h5'
dbPath = 'data/database'

#device
cuda = torch.cuda.is_available()
if cuda:
    device =  torch.device("cuda:0")
else:
    device =  torch.device("cpu")
    print("No GPU found, please get one")
    
#model
encoder = models.vgg16(weights=None)
encoder_k = 512 ##TODO
layers = list(encoder.features.children())[:-2]
model = nn.Module()
encoder = nn.Sequential(*layers)
model.add_module('encoder', encoder)
n_vocabs = 16
net_vlad = NetVLADLayer(n_vocabs = n_vocabs, k = encoder_k)
model.add_module('netvlad', net_vlad)
model = model.to(device)




@app.post("/query_one", response_model = query_one)
async def fas(file: UploadFile = File()):
    file_path = './data/'+ file.filename
    file_result_path = "./data/result.jpg"
    with open(file_path, 'wb') as f:
        f.write(file.file.read())
    
    image = read_image(file_path)
    retrived_dict = query_one(image, device, model, db_features = db_features)
    db_imgs = [read_image(Path(dbPath) / r) for r in retrived_dict]
    
    return db_imgs
       
if __name__ == '__main__' : 
    uvicorn.run(app,host="0.0.0.0",port=2014)
