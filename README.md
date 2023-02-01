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
Online triplet mining refer to taken from a **batch** a triplet of [anchor, positive, negative] so that the triplet loss is **largest**.
Then use that triplet to train the model.

### Problem statement
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
- The problem is worse when there isn't sufficient number of hard sample in trainning set; As the probability of hitting a hard pair is very low.
### Online Triplet Mining
Aware of such problem, a technique called triplet mining was developed. In a nutshell, Triplet Mining refer to calculation triplet pairs **before** selecting approximate triplet to train the model.
Such approach is called Offline Triplet Mining, which requires the computation of embedding of all data before training. Therefore, requires a lot of computation resources.

Instead of Offline Triplet Mining, A. Hermans, L. Beyers and Bastian Leibe (2017) proposed the use of Online Triplet Mining. The main different in Online Triplet Mining is that the embedding is caclulated **at run time** for the current batch. From which, suitable triplets are selected.
### Batch Hard
For each anchor:
 - Take the hardest positive
 - Take the hardest negative
 - calculate triplet loss
Average over this set of **hardest** triplet loss
### Batch All
For each anchor:
- Take all positive
- Take all negative
- Calculate all triplet losses
Remove losses whose value equals zero
Average over the remaining losses to get the triplet loss.
## Why Batch All then Batch Hard
Experiment shown that using Batch Hard from the beginning lead to model collapse (i.e all point mapped to the same position in embedding space).
Whereas using naive Triplet leads to suboptimal results.
-> Use Batch All till convergence, then use Batch Hard for **some** epochs.