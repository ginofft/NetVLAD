import numpy as np

def train(epoch):
    epoch_loss = 0
    startIter = 1

    subsetN = 1
    subsetIdx = [np.arrange(len(train_set)), subsetN]
     