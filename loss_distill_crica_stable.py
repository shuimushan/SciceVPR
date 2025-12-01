from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
import torch
from torch.nn import functional as F
loss_fn = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())
loss_l2 = torch.nn.MSELoss(reduction='sum')
#import numpy as np
#criterion = torch.nn.MSELoss()
#criterion = torch.nn.L1Loss()
    
    
#  The loss function call (this method will be called at each training iteration)
def loss_function(descriptors, descriptors_crica, labels):
    # we mine the pairs/triplets if there is an online mining strategy
    if miner is not None:

        miner_outputs = miner(descriptors, labels)
        loss_global = loss_fn(descriptors, labels, miner_outputs)
        loss_distill = loss_l2(descriptors, descriptors_crica)/descriptors_crica.shape[0]#不更新crica只更新distill，crica值只受loss_crica影响
        # calculate the % of trivial pairs/triplets 
        # which do not contribute in the loss value
        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined/nb_samples)

    else: # no online mining
        loss = loss_fn(global_descriptors, labels)
        batch_acc = 0.0
    return loss_global, loss_distill
