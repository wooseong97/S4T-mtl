# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

from shutil import ignore_patterns
import warnings
import cv2
import os.path
import numpy as np
import glob
import torch
import json
import scipy.io as sio


def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred


class ScumMeter(object):
    def __init__(self, ignore_index=-10000, task='depth', normalize=False):
        self.total_rmses = 0.0
        self.total_log_rmses = 0.0
        self.n_valid = 0.0
        self.ignore_index = ignore_index

        self.abs_rel = 0.0
        self.sq_rel = 0.0
        
        self.task = task
        self.normalize = normalize

    @torch.no_grad()
    def update(self, pred, gt):    
        pred, gt = pred.squeeze(), gt.squeeze()
        
        # Determine valid mask
        mask = (gt != self.ignore_index).bool()
        self.n_valid += mask.float().sum().item() # Valid pixels per image
        
        if self.normalize:
            gt[mask] = gt[mask]/self.normalize
            pred[mask] = pred[mask]/self.normalize
        
        # alignment
        pred, scale, shift = align_depth_least_square(
            gt_arr=gt.cpu().numpy(),
            pred_arr=pred.cpu().numpy(),
            valid_mask_arr=mask.cpu().numpy(),
            return_scale_shift=True,
            max_resolution=None,
        )
        pred = torch.from_numpy(pred).cuda()

        # Per pixel rmse and log-rmse.
        log_rmse_tmp = torch.pow(torch.log(gt[mask]) - torch.log(pred[mask]), 2)
        self.total_log_rmses += log_rmse_tmp.sum().item()

        rmse_tmp = torch.pow(gt[mask] - pred[mask], 2)
        self.total_rmses += rmse_tmp.sum().item()

        # abs rel
        self.abs_rel += torch.abs(torch.abs(gt[mask] - pred[mask])).sum().item()
        # sq_rel
        self.sq_rel += (((gt[mask] - pred[mask]) ** 2) / gt[mask]).sum().item()

    def reset(self):
        self.rmses = []
        self.log_rmses = []
        
    def get_score(self, verbose=True):
        eval_result = dict()
        eval_result['rmse'] = np.sqrt(self.total_rmses / self.n_valid)
        eval_result['log_rmse'] = np.sqrt(self.total_log_rmses / self.n_valid)
        eval_result['l1'] = self.abs_rel / self.n_valid
        eval_result['sq_rel'] = self.sq_rel / self.n_valid

        if verbose:
            print('Results for %s'%self.task)
            for x in eval_result:
                spaces = ''
                for j in range(0, 15 - len(x)):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_result[x]))

        return eval_result
