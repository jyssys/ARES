import torch
import torch.nn.functional as F

import numpy as np

'''
manifold mixup
'''

def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    out = out*lam + out[indices]*(1-lam)
    
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
    
    return out, target_reweighted

# def mixup_process(out, target_reweighted, lam):
#     indices = np.random.permutation(out.size(0))
#     out = out/(1-lam)-lam*out[indices]/(1-lam)
    
#     target_shuffled_onehot = target_reweighted[indices]
#     target_reweighted = target_reweighted/(1-lam)-lam*target_shuffled_onehot/(1-lam)
    
#     return out, target_reweighted

def to_one_hot(inp,num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    
    #return Variable(y_onehot.cuda(),requires_grad=False)
    return y_onehot.cuda()

def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam