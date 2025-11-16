from itertools import permutations

import math
import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.nn import functional as F

from .utils import dequeue_and_enqueue

def compute_unsupervised_loss(predict, target):
    batch_size, num_class, h, w = predict.shape
    weight = batch_size * h * w / torch.sum(target != 255)

    loss = weight * F.cross_entropy(predict, target, ignore_index=255)  # [10, 321, 321]

    return loss

def compute_corr_loss(pred_corr, target):

    loss = F.cross_entropy(pred_corr, target.long(), ignore_index=255)  # [10, 321, 321]

    return loss


def compute_rank_loss_landmark(feat_x1, feat_x2, feat_x3, high_feat, corr_map, label):
    criterion_c = torch.nn.KLDivLoss(reduction='mean').cuda() # pixel-reference correlation criterion

    num_landmarks = 64

    refers_h, refers_f1, refers_f2, refers_f3 = orthogonal_landmarks_refine(feat_x1, feat_x2, feat_x3, high_feat, corr_map, label, num_landmarks)
    B, D, H, W = high_feat.shape
    feat_x1 = F.interpolate(feat_x1.detach(), (H, W), mode='bilinear', align_corners=True)
    feat_x2 = F.interpolate(feat_x2.detach(), (H, W), mode='bilinear', align_corners=True)
    feat_x3 = F.interpolate(feat_x3.detach(), (H, W), mode='bilinear', align_corners=True)
    p2r_h = torch.einsum('b c h w, b n c -> b h w n', high_feat, refers_h).softmax(dim=-1)
    p2r_f1 = torch.einsum('b c h w, b n c -> b h w n', feat_x1, refers_f1).softmax(dim=-1)
    p2r_f2 = torch.einsum('b c h w, b n c -> b h w n', feat_x2, refers_f2).softmax(dim=-1)
    p2r_f3 = torch.einsum('b c h w, b n c -> b h w n', feat_x3, refers_f3).softmax(dim=-1)

    loss = (criterion_c((p2r_f1 + 1e-10).log(), p2r_f2) + criterion_c((p2r_f2 + 1e-10).log(), p2r_f3) + criterion_c((p2r_f3 + 1e-10).log(), p2r_h)) / 3

    return loss

def orthogonal_landmarks_refine(q1, q2, q3, qh, corr_map, label, num_landmarks=64):
    B, Dh, Hh, Wh = qh.shape
    Nh = Hh * Wh
    _, D1, Hl, Wl = q1.shape
    _, D2, _, _ = q2.shape
    _, D3, _, _ = q3.shape
    Nl = Hl * Wl
    L2H = math.ceil(Wl / Wh)
    qh = qh.permute(0, 2, 3, 1).reshape(B, -1, Dh)
    q1 = q1.permute(0, 2, 3, 1).reshape(B, -1, D1)
    q2 = q2.permute(0, 2, 3, 1).reshape(B, -1, D2)
    q3 = q3.permute(0, 2, 3, 1).reshape(B, -1, D3)

    label = F.interpolate(label.unsqueeze(1).float(), (Hh, Wh), mode="nearest").squeeze(1).to(torch.int)
    label = label.reshape(B, -1)

    selected_mask_h = torch.zeros((B, Nh, 1), device=qh.device)
    selected_mask_l = torch.zeros((B, Nl, 1), device=qh.device)
    landmark_mask = torch.ones((B, 1, 1), dtype=selected_mask_h.dtype, device=qh.device)
    random_idx = torch.zeros((B, 1, 1), dtype=torch.int64, device=qh.device)

    for b in range(B):
        indices = torch.nonzero(label[b])
        if indices.size(0) > 0:
            random_index = indices[torch.randint(0, indices.size(0), (1,))]
            random_idx[b] = random_index
        else:
            random_index = torch.randint(label.size(-1), (1, 1))
            random_idx[b] = random_index
    random_idx_l = (random_idx//Wh)*L2H*Wl+(random_idx%Wh)*L2H
    selected_landmark_h = random_idx.view(B, 1)
    selected_landmark_l = random_idx_l.view(B, 1)
    selected_mask_h.scatter_(-2, random_idx, landmark_mask)
    selected_mask_l.scatter_(-2, random_idx_l, landmark_mask)

    # Selected landmarks
    selected_landmarks = torch.empty((B, num_landmarks, 1), device=qh.device, dtype=random_idx.dtype)
    selected_landmarks[:, 0, :] = selected_landmark_h

    sims = torch.empty((B, Nh, num_landmarks), device=qh.device, dtype=qh.dtype)
    for b in range(B):
        selected_mask_h_b = selected_mask_h[b].unsqueeze(0)
        selected_mask_l_b = selected_mask_l[b].unsqueeze(0)
        for M in range(1, num_landmarks):
            sim = corr_map[b, selected_landmark_h[b], :]
            sims[b, :, M - 1] = sim
            if M>1:
                sims[b, :, M - 1] += sims[b, :, M - 2]
            sim_set = sims[b, :, :M]
            sim_set.view(-1, M)[selected_mask_h_b.flatten().bool(), :] = 100000
            selected_landmark_h[b] = sim_set.amax(-1).argmin(-1)
            selected_landmark_l[b] = (selected_landmark_h[b]//Wh)*L2H*Wl+(selected_landmark_h[b]%Wh)*L2H
            selected_landmarks[b, M, :] = selected_landmark_h[b]

            selected_mask_h_b.scatter_(-2, selected_landmark_h[b].unsqueeze(-1).unsqueeze(-1), landmark_mask)
            selected_mask_l_b.scatter_(-2, selected_landmark_l[b].unsqueeze(-1).unsqueeze(-1), landmark_mask)
        selected_mask_h[b] = selected_mask_h_b[0, :, :]
        selected_mask_l[b] = selected_mask_l_b[0, :, :]

    landmarks_h = torch.masked_select(qh, selected_mask_h.bool()).reshape(B, -1, Dh)  # (B, M, D)
    landmarks_1 = torch.masked_select(q1, selected_mask_l.bool()).reshape(B, -1, D1)
    landmarks_2 = torch.masked_select(q2, selected_mask_h.bool()).reshape(B, -1, D2)
    landmarks_3 = torch.masked_select(q3, selected_mask_h.bool()).reshape(B, -1, D3)
    return landmarks_h, landmarks_1, landmarks_2, landmarks_3

def prob2rank(prob, prob_s, k=4):
    """
    input: prob(probability) [b, h, w, n]
    return: rank [b, h, w, k!]
    To save the computing resources, use top-k ranther than n
    """
    full_permutation = [c for c in permutations(range(k))]
    full_permutation = torch.from_numpy(np.stack(full_permutation)) # [k!, k]

    _, prob_topk_index = prob.topk(k, dim=-1) # [b, h, w, k]
    A = prob_topk_index[:, :, :, full_permutation] # [b, h, w, k!, k]
    B = prob.unsqueeze(3).expand(-1, -1, -1, full_permutation.shape[0], -1) # [b, h, w, k!, n]
    B_s = prob_s.unsqueeze(3).expand(-1, -1, -1, full_permutation.shape[0], -1)
    C = torch.gather(input=B, dim=-1, index=A) # [b, h, w, k!, k]
    C_s = torch.gather(input=B_s, dim=-1, index=A)

    rank = C[:, :, :, :, 0] / (C[:, :, :, :, 0:].sum(dim=-1) + 1e-10) # [b, h, w, k!]
    rank_s = C_s[:, :, :, :, 0] / (C_s[:, :, :, :, 0:].sum(dim=-1) + 1e-10)

    for i in range(1, k):
        rank *= C[:, :, :, :, i] / (C[:, :, :, :, i:].sum(dim=-1) + 1e-10)
        rank_s *= C_s[:, :, :, :, i] / (C_s[:, :, :, :, i:].sum(dim=-1) + 1e-10)

    return rank, rank_s

def prob2ranks(prob, prob_1, prob_2, prob_3, k=4):
    """
    input: prob(probability) [b, h, w, n]
    return: rank [b, h, w, k!]
    To save the computing resources, use top-k ranther than n
    """
    full_permutation = [c for c in permutations(range(k))]
    full_permutation = torch.from_numpy(np.stack(full_permutation)) # [k!, k]

    _, prob_topk_index = prob.topk(k, dim=-1) # [b, h, w, k]
    A = prob_topk_index[:, :, :, full_permutation] # [b, h, w, k!, k]
    B = prob.unsqueeze(3).expand(-1, -1, -1, full_permutation.shape[0], -1) # [b, h, w, k!, n]
    B1_s = prob_1.unsqueeze(3).expand(-1, -1, -1, full_permutation.shape[0], -1)
    B2_s = prob_2.unsqueeze(3).expand(-1, -1, -1, full_permutation.shape[0], -1)
    B3_s = prob_3.unsqueeze(3).expand(-1, -1, -1, full_permutation.shape[0], -1)
    C = torch.gather(input=B, dim=-1, index=A) # [b, h, w, k!, k]
    C1_s = torch.gather(input=B1_s, dim=-1, index=A)
    C2_s = torch.gather(input=B2_s, dim=-1, index=A)
    C3_s = torch.gather(input=B3_s, dim=-1, index=A)

    rank = C[:, :, :, :, 0] / (C[:, :, :, :, 0:].sum(dim=-1) + 1e-10) # [b, h, w, k!]
    rank1_s = C1_s[:, :, :, :, 0] / (C1_s[:, :, :, :, 0:].sum(dim=-1) + 1e-10)
    rank2_s = C2_s[:, :, :, :, 0] / (C2_s[:, :, :, :, 0:].sum(dim=-1) + 1e-10)
    rank3_s = C3_s[:, :, :, :, 0] / (C3_s[:, :, :, :, 0:].sum(dim=-1) + 1e-10)

    for i in range(1, k):
        rank *= C[:, :, :, :, i] / (C[:, :, :, :, i:].sum(dim=-1) + 1e-10)
        rank1_s *= C1_s[:, :, :, :, i] / (C1_s[:, :, :, :, i:].sum(dim=-1) + 1e-10)
        rank2_s *= C2_s[:, :, :, :, i] / (C2_s[:, :, :, :, i:].sum(dim=-1) + 1e-10)
        rank3_s *= C3_s[:, :, :, :, i] / (C3_s[:, :, :, :, i:].sum(dim=-1) + 1e-10)

    return rank, rank1_s, rank2_s, rank3_s


def get_criterion(cfg):
    cfg_criterion = cfg["criterion"]
    aux_weight = (
        cfg["net"]["aux_loss"]["loss_weight"]
        if cfg["net"].get("aux_loss", False)
        else 0
    )
    ignore_index = cfg["dataset"]["ignore_label"]
    if cfg_criterion["type"] == "ohem":
        criterion = CriterionOhem(
            aux_weight, ignore_index=ignore_index, **cfg_criterion["kwargs"]
        )
    else:
        criterion = Criterion(
            aux_weight, ignore_index=ignore_index, **cfg_criterion["kwargs"]
        )

    return criterion


class Criterion(nn.Module):
    def __init__(self, aux_weight, ignore_index=255, use_weight=False):
        super(Criterion, self).__init__()
        self._aux_weight = aux_weight
        self._ignore_index = ignore_index
        self.use_weight = use_weight
        if not use_weight:
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            weights = torch.FloatTensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ).cuda()
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
            self._criterion1 = nn.CrossEntropyLoss(
                ignore_index=ignore_index, weight=weights
            )

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                    len(preds) == 2
                    and main_h == aux_h
                    and main_w == aux_w
                    and main_h == h
                    and main_w == w
            )
            if self.use_weight:
                loss1 = self._criterion(main_pred, target) + self._criterion1(
                    main_pred, target
                )
            else:
                loss1 = self._criterion(main_pred, target)
            loss2 = self._criterion(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion(preds, target)
        return loss


class CriterionOhem(nn.Module):
    def __init__(
            self,
            aux_weight,
            thresh=0.7,
            min_kept=100000,
            ignore_index=255,
            use_weight=False,
    ):
        super(CriterionOhem, self).__init__()
        self._aux_weight = aux_weight
        self._criterion1 = OhemCrossEntropy2dTensor(
            ignore_index, thresh, min_kept, use_weight
        )
        self._criterion2 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                    len(preds) == 2
                    and main_h == aux_h
                    and main_w == aux_w
                    and main_h == h
                    and main_w == w
            )

            loss1 = self._criterion1(main_pred, target)
            loss2 = self._criterion2(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion1(preds, target)
        return loss


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0 / factor, 1.0 / factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0 / factor, 1.0 / factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (
                factor * factor
        )  # int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept) - 1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = (
            torch.from_numpy(input_label.reshape(target.size()))
            .long()
            .cuda(target.get_device())
        )

        return new_target

    def forward(self, predict, target, weight=None):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)


class OhemCrossEntropy2dTensor(nn.Module):
    """
    Ohem Cross Entropy Tensor Version
    """

    def __init__(
            self, ignore_index=255, thresh=0.7, min_kept=256, use_weight=False, reduce=False
    ):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [
                    0.8373,
                    0.918,
                    0.866,
                    1.0345,
                    1.0166,
                    0.9969,
                    0.9754,
                    1.0489,
                    0.8786,
                    1.0023,
                    0.9539,
                    0.9843,
                    1.1116,
                    0.9037,
                    1.0865,
                    1.0955,
                    1.0865,
                    1.1529,
                    1.0507,
                ]
            ).cuda()
            # weight = torch.FloatTensor(
            #    [0.4762, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
            #    1.4286, 0.5, 3.3333,5.0, 10.0, 2.5, 0.8333]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=weight, ignore_index=ignore_index
            )
        elif reduce:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=ignore_index
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_index
            )

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            # print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)
