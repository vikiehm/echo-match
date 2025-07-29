# Loss functions inherited from original DPFM repo with some modifications (https://github.com/pvnieo/DPFM)
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.registry import LOSS_REGISTRY

class FrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        loss = torch.sum((a - b) ** 2, axis=(1, 2))
        return torch.mean(loss)

class WeightedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.binary_loss = nn.BCELoss(reduction="none")

    def forward(self, prediction, gt):
        class_loss = self.binary_loss(prediction, gt)

        # Handle degenerate cases where gt is all 0s or all 1s
        n_positive = gt.sum()
        total = gt.size(0)
        if n_positive == 0 or n_positive == total:
            # Note: When gt contains only one class, reweighting would zero out the loss.
            # Fallback to unweighted BCE loss in these cases.
            return torch.mean(class_loss)

        weights = torch.ones_like(gt)
        w_negative = n_positive / total
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative

        return torch.mean(weights * class_loss)

class NCESoftmaxLoss(nn.Module):
    def __init__(self, nce_t, nce_num_pairs=None):
        super().__init__()
        self.nce_t = nce_t
        self.nce_num_pairs = nce_num_pairs
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, feat_x, feat_y, corr_x, corr_y): #  we've changed orignal map21 to corr_x and corr_y to take into accound partial to partial
        # don't consider batch for ease of implementation

        feat_x, feat_y = feat_x.squeeze(0), feat_y.squeeze(0)
        feat_x, feat_y = F.normalize(feat_x, p=2, dim=-1), F.normalize(feat_y, p=2, dim=-1)

        logits = feat_x @ feat_y.transpose(0,1) / self.nce_t   # Nx x Ny

        logits_x = logits[corr_x]
        logits_y = logits.transpose(0,1)[corr_y]

        loss_x = F.cross_entropy(logits_x, corr_y)
        loss_y = F.cross_entropy(logits_y, corr_x)

        return loss_x + loss_y

@LOSS_REGISTRY.register()
class EchoMatchLoss(nn.Module):
    def __init__(self, w_fmap=1, w_acc=1, w_nce_self=1, w_nce_cross=0.1, nce_t=0.07, nce_num_pairs=4096, **kwargs):
        super().__init__()

        self.w_fmap = w_fmap
        self.w_acc = w_acc
        self.w_nce_self = w_nce_self
        self.w_nce_cross = w_nce_cross

        self.frob_loss = FrobeniusLoss()
        self.binary_loss = WeightedBCELoss()
        self.nce_softmax_loss = NCESoftmaxLoss(nce_t, nce_num_pairs)

    def forward(self, C12, C_gt, corr_x, corr_y, feat1, feat2, overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21): # we've changed map21 to corr_x and corr_y
        loss = 0

        # fmap loss
        fmap_loss = self.frob_loss(C12, C_gt) * self.w_fmap
        loss += fmap_loss



        # overlap loss
        acc_loss = self.binary_loss(overlap_score12, gt_partiality_mask12.float()) * self.w_acc
        acc_loss += self.binary_loss(overlap_score21, gt_partiality_mask21.float()) * self.w_acc

        loss += acc_loss

        # nce loss
        cross_nce_loss = self.nce_softmax_loss(feat1, feat2, corr_x, corr_y)

        feat1, feat2 = feat1.squeeze(0), feat2.squeeze(0)
        feat1, feat2 = F.normalize(feat1, p=2, dim=-1), F.normalize(feat2, p=2, dim=-1)
        logits_x_self = feat1 @ feat1.transpose(0,1) / 0.07
        logits_y_self = feat2 @ feat2.transpose(0,1) / 0.07
        lablels_x = torch.arange(feat1.shape[0]).long().to(feat1.device)
        lablels_y = torch.arange(feat2.shape[0]).long().to(feat2.device)
        loss_x_self = F.cross_entropy(logits_x_self, lablels_x)
        loss_y_self = F.cross_entropy(logits_y_self, lablels_y)
        self_nce_loss = loss_x_self + loss_y_self
        
        final_nce_loss = cross_nce_loss * self.w_nce_cross + self_nce_loss * self.w_nce_self

        loss += final_nce_loss

        return fmap_loss, acc_loss, final_nce_loss
