# Criterions
import torch 
import torch.nn as nn 
from torch.nn import functional as F


def ce_loss(logits, targets, reduction='mean'):
    # cross entropy loss in pytorch.
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)

def consistency_loss(logits, target, loss_type='ce', mask=None, disagree_weight_masked=None):
    # consistency loss
    # input: logits_x_ulb_s, pseudo_label, loss_type, mask
    if loss_type == 'mse':
        loss = F.mse_loss(logits, target, reduction='none')
    else:
        loss = ce_loss(logits, target, reduction='none')

    if mask is not None:
        # add pseudo-label mask
        loss = loss * mask
        if disagree_weight_masked is not None:
            # add disagreement weight and mask
            loss = loss * disagree_weight_masked

    return loss.mean()