# Pytorch NetVLAD with Online Triplet Mining
Pytorch implementation of NetVLAD with Online Triplet Mining (Batch Hard and Batch All)
## References
* **Relja Arandjelovic et al**. *NetVLAD: CNN architecture for weakly supervised place recognition (2015)*. https://doi.org/10.48550/arXiv.1511.07247 
* **Alexander Hermans, Lucas Beyer and Bastian Leibe**. *In Defense of the Triplet Loss for Person Re-Identification (2017)*. https://doi.org/10.48550/arXiv.1511.07247

## Quick Start
To make use of Online triplet mining:
- train with batchAll till convergence
- train with batchHard for **some** epochs.
  
The train and validation folder format is as followed:  
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
To train (initially):
```
python main.py --mode train --tripletLoss batchall --nEpochs 1000 \
  --trainPath data/train \
  --validationPath data/validation \
  --savePath model/BatchAll \
```
Switching loss function into BatchHard, then train:
```
python main.py --mode train --tripletLoss batchhard --nEpochs 1000 \
  --trainPath data/train \
  --validationPath data/validation \
  --savePath model/batchHard \
  --loadPath model/BatchAll/best.pth.tar
  --oldLoss False # This argument is needed when switching Loss function
```
To inference:
```
python main.py --mode test \
  --dbPath data/database \
  --queryPath data/query \
  --loadPath model/BatchAll/best.pth.tar \
  --outPath out
```

## Online Triplet Mining: Batch Hard and Batch All
Naive triplet loss made use of random triplets, which usually resulted in:
- slow convergence
- loss stuck in local minima
The root cause is due to how triplets are constructed - a visualization is shown below.
![tripletPairs](https://user-images.githubusercontent.com/80506834/212865622-5ea29b61-3abd-4751-be50-452fb4823457.png)

Given a random anchor, we can construct the following pairs:
- Easy Positive (Easy Pos): same class; Is near the anchor
- Hard Positive (Hard Pos): same class; Is far from the anchor
- Easy Negative (Easy Neg): different class; Is near the anchor
- Hard Negative (Hard Neg): different class; Is far from the anchor

Triplet loss calculate the loss given by a [anchor, positive, negative] triplet, however, not all triplet ares equal!!
- If [anchor, positive] and [anchor, negative] sastify the conditions of easy pos/neg -> There is **nothing** to learn from this triplet

=> We need to take [anchor, positive, negative] triplet; That create both: Hard Pos and Hard Neg pairs



