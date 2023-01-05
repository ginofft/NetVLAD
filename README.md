# Pytorch NetVLAD with Online Triplet Mining
Pytorch implementation of NetVLAD with Online Triplet Mining (Batch Hard and Batch All)
## References
* **Relja Arandjelovic et al**. *NetVLAD: CNN architecture for weakly supervised place recognition (2015)*. https://doi.org/10.48550/arXiv.1511.07247 
* **Alexander Hermans, Lucas Beyer and Bastian Leibe**. *In Defense of the Triplet Loss for Person Re-Identification (2017)*. https://doi.org/10.48550/arXiv.1511.07247

## Quick Start
The model for this dataset is stored [here](https://drive.google.com/file/d/1ZurYnT9hw9KRl2fLyNAJfTCmlw0OdUGa/view?usp=sharing).  
Download and put the model inside `/model/BatchAll/`.  
The train and validation format is as followed:  
```
.
├── train            
│   ├── class1
│   │   ├──img1_c1.jpg
│   │   ├──img2_c1.png
│   │   ├──...
│   ├── class2
│   │   ├──img1_c2.jpg
│   │   ├──img2_c2.png
│   │   ├──...
```
To train:
```
python main.py --mode train --tripletLoss batchall --nEpochs 1000 \
  --trainPath /data/train \
  --validationPath /data/validation \
  --savePath /model/BatchAll \
  --loadPath /model/BatchAll/demo_version.pth.tar #Optional(if train from resume)
```
To inference:
```
python main.py --mode test \
  --dbPath /data/database \
  --queryPath /data/query \
  --loadPath /model/BatchAll/demo_version.pth.tar \
  --outPath /out
```
