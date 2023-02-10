import streamlit as st
from pathlib import Path
import sys
import requests
import os
import json
import numpy as np
import cv2
import glob
import numpy as np
import base64
import random
from streamlit_image_select import image_select
from PIL import Image

# API_query_one = 'http://172.18.5.30:2014/query_one'


PATH_DATASET = "/storage/computervision/longnth/models/image_retrieval/NetVLAD/data/database"


def select_image(images,captions=None):
    img = image_select(
    label="Select a photo",
    images=images,
    captions=captions,
    use_container_width=False
    )
    return img


    
list_label_structure = ["british_museum", "florence_cathedral_side", "lincoln_memorial", "milan_cathedral", "mount_rushmore", "piazza_san_macro", "reichstag", "sacre_coeur", "sangrada_familia", "st_pauls_cathedral", "st_peters_square"]
list_dataset = ["train", "validation"]
dict_att = {}

number_img_sample = 10

sys.path.append('..')
st.set_page_config(
    page_title="Image Retrieval",layout='wide'
)
st.markdown("<h1 style='text-align: center; color: white;'>Image Retrieval</h1>", unsafe_allow_html=True)

# att = os.listdir("/storage/computervision/longnth/models/image_retrieval/NetVLAD/data/database")
att = os.listdir("/storage/computervision/longnth/models/image_retrieval/NetVLAD/data")


# --------- query_one ---------------
# import sys
# # caution: path[0] is reserved for script path (or '' in REPL)
# sys.path.insert(1, '/storage/computervision/longnth/models/image_retrieval/NetVLAD')
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import VGG16_Weights

import importlib.util

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NetVLADLayer = module_from_file("NetVLADLayer", "/storage/computervision/longnth/models/image_retrieval/NetVLAD/scr/netvlad.py")
load_checkpoint = module_from_file("load_checkpoint", "/storage/computervision/longnth/models/image_retrieval/NetVLAD/scr/utils.py")
query = module_from_file("query", "/storage/computervision/longnth/models/image_retrieval/NetVLAD/scr/query.py")

cuda = torch.cuda.is_available()
if cuda:
    device =  torch.device("cuda")
else:
    device =  torch.device("cpu")
    
encoder = models.vgg16(weights=None)
encoder_k = 512 ##TODO
layers = list(encoder.features.children())[:-2]

model = nn.Module()
encoder = nn.Sequential(*layers)
model.add_module('encoder', encoder)

n_vocabs = 16
net_vlad = NetVLADLayer.NetVLADLayer(n_vocabs = n_vocabs, k = encoder_k)
model.add_module('netvlad', net_vlad)

model = model.to(device)


#Set up output paths
dbPath = '/storage/computervision/longnth/models/image_retrieval/NetVLAD/data/database'
queryPath = '/storage/computervision/longnth/models/image_retrieval/NetVLAD/data/query'
outPath = '/storage/computervision/longnth/models/image_retrieval/NetVLAD/out'
query_features = Path(outPath) / 'q_features.h5'
db_features = Path(outPath) / 'db_features.h5'
retrieval = Path(outPath) / 'retrieved.h5'

# db_dataset = ImageDataset(Path(dbPath))
loadPath = '/storage/computervision/longnth/models/image_retrieval/NetVLAD/model/BatchAll/best.pth.tar'

startEpoch, train_loss, val_loss = load_checkpoint.load_checkpoint(Path(loadPath), 
                                                        device,
                                                        model)

query_one = module_from_file("query_one", "/storage/computervision/longnth/models/image_retrieval/NetVLAD/scr/query_one.py")
# plot_retrieval_images_one = module_from_file("plot_retrieval_images_one", "/storage/computervision/longnth/models/image_retrieval/NetVLAD/scr/query_one.py")

# API_query_one = query_one.query_one(pil_image, device, model,
#                       db_features,
#                       n_result=10)

# --------- process label ---------------
st.write("### Upload query image")
source = st.file_uploader("Choose a file")

if source is not None : 
    file = Path('Data/014_Image Retrieval/Uploaded_Image/source.jpg')
    content = source.read()
    file.write_bytes(content)
    st.image('Data/014_Image Retrieval/Uploaded_Image/source.jpg',caption = 'Successful upload' ,use_column_width = 'auto')
else:        
    option_source_dataset = st.selectbox(
        'Selected your dataset',
        list_dataset)

    option_source_label = st.selectbox(
        'Selected your structure label',
        list_label_structure)
    db_dict = glob.glob(f"/storage/computervision/longnth/models/image_retrieval/NetVLAD/data/{option_source_dataset}/{option_source_label}/*")[:number_img_sample]
    img = select_image(images = db_dict)
    file = str(img)
if st.button('Find Similars') :
    with st.spinner("Processing..."):
        # call api
        files = {'file': open(file, 'rb')}
#         response = requests.post(API_query_one, files=files)
        retrieved_dict = query_one.query_one(Image.open(file), device, model, db_features, n_result=10)
#         plot_retrieval_images_one.plot_retrieval_images_one(query_img = Image.open(file), retrieved_dict = retrieved_dict, db_dir = dbPath)
        st.image(Image.open(file))
# #         result = response.json()
# #         response = list(response.values())[0]  
        col1,col2,col3,col4 = st.columns(4)
        cols = [col1,col2,col3,col4]
        i = 0
        for similar_image in retrieved_dict :
            j = i%4
            i+=1
            with cols[j] : 
                encode = Image.open(dbPath+"/"+similar_image)
                open_cv_image = cv2.cvtColor(np.array(encode), cv2.COLOR_RGB2BGR)
#                 im_bytes = base64.b64decode(encode)
#                 im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
#                 img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
                img = cv2.resize(open_cv_image, (300,300))
                st.image(img[:,:,::-1],caption ='Similar' ,use_column_width = 'auto')
