#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.AutomaticWeightedLoss import AutomaticWeightedLoss


def compute_task_loss(outputs, targets, task):
    """ Functional per-task loss used for the Taskonomy source domain. """
    if 'class' in task:
        task_loss = F.mse_loss(outputs, targets.squeeze(1))
    elif 'segment_semantic' in task:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        task_loss = criterion(outputs, targets.long())
    elif 'normal' in task:
        T = targets
        task_loss = (1 - (outputs*T).sum(-1) / (torch.norm(outputs, p=2, dim=-1) + 0.000001) / (torch.norm(T, p=2, dim=-1)+ 0.000001) ).mean()
    elif 'depth' in task or 'keypoint' in task or 'reshading' in task or 'edge' in task or 'segment' in task:
        if len(outputs.shape) == 4:
            Out = outputs.squeeze()
        task_loss = F.l1_loss(Out, targets)
    else: # L2 curvature
        if len(outputs.shape) == 4:
            Out = outputs.squeeze()
        task_loss = F.mse_loss(Out, targets)
    return task_loss


def seg_tasks(tasks):
    if 'semseg' in tasks: return 'semseg'
    elif 'segment_semantic' in tasks: return 'segment_semantic'
    else: return False

def seg_zero_detect(gt, sem_target):
    if not sem_target: return False
    elif torch.sum(gt[sem_target]==255) == torch.numel(gt[sem_target]): return True
    else: return False

def out_loss(pred, gt, loss_ft, task, seg_target, sem_zero):
    if sem_zero and task==seg_target: return torch.zeros(1, dtype=torch.float)[0].cuda()
    else: return loss_ft(pred, gt)

def out_loss2(pred, gt, task, seg_target, sem_zero):
    if sem_zero and task==seg_target: return torch.zeros(1, dtype=torch.float)[0].cuda()
    else: return compute_task_loss(pred, gt, task)


class MTLoss_S4T_v2(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict, ttt: bool):
        super(MTLoss_S4T_v2, self).__init__()
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        self.ttt = ttt
        self.seg_target = seg_tasks(tasks)
        if 'aux_scale' in loss_weights.keys():
            self.aux_scale=True
            self.aux_tasks = self.p.AUX_TASKS.NAMES
            self.AWL2 = AutomaticWeightedLoss(len(self.aux_tasks))
        else: self.aux_scale=False

    def forward(self, pred_pre, pred_joint, aux_pred, gt, tasks):
        sem_zero = seg_zero_detect(gt, self.seg_target)

        out = {}
        for task in tasks:
            out['pre_'+task] = out_loss(pred_pre[task], gt[task], self.loss_ft[task], task, self.seg_target, sem_zero)*self.loss_weights[task]
            out['aux_'+task] = out_loss(aux_pred[task], gt[task], self.loss_ft[task], task, self.seg_target, sem_zero)*self.loss_weights[task]
            out['S4T_'+task+'_pretrain']  = out_loss(pred_joint[task], gt[task], self.loss_ft[task], task, self.seg_target, sem_zero)*self.loss_weights[task]
        out['total_pre'] = torch.sum(torch.stack([out['pre_'+t] for t in tasks]))
        out['total_aux'] = torch.sum(torch.stack([out['aux_'+t] for t in tasks]))* self.loss_weights['TTT_loss']
        out['total_S4T_pretrain'] = torch.sum(torch.stack([out['S4T_'+t+'_pretrain'] for t in tasks]))* self.loss_weights['TTT_loss']
        out['total'] = out['total_pre'] +out['total_aux'] + out['total_S4T_pretrain']

        if self.ttt:
            for task in tasks:
                out['S4T_'+task+'_ttt'] = out_loss(pred_joint[task], pred_pre[task].detach(), self.loss_ft[self.p.TTT_loss], task, self.seg_target, sem_zero)*self.loss_weights[task]
            out['total_S4T_ttt'] = torch.sum(torch.stack([self.loss_weights[t] * out['S4T_'+t+'_ttt'] for t in tasks]))* self.loss_weights['TTT_loss']
            out['total'] = out['total_S4T_ttt']

        return out


class MTLoss_S4T_taskonomy_v2(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict, ttt: bool):
        super(MTLoss_S4T_taskonomy_v2, self).__init__()
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        self.AWL = AutomaticWeightedLoss(len(p.TASKS.NAMES))
        self.ttt = ttt
        self.seg_target = seg_tasks(tasks)
        if 'aux_scale' in loss_weights.keys():
            self.aux_scale=True
            self.aux_tasks = self.p.AUX_TASKS.NAMES
            self.AWL2 = AutomaticWeightedLoss(len(self.aux_tasks))
        else: self.aux_scale=False


    def forward(self, pred_pre, pred_joint, aux_pred, gt, tasks):
        sem_zero = seg_zero_detect(gt, self.seg_target)

        out = {}
        loss_pre_list, loss_aux_list, loss_TTT_list = [], [], []
        for task in tasks:
            out['pre_'+task] = out_loss2(pred_pre[task], gt[task], task, self.seg_target, sem_zero)
            out['aux_'+task] = out_loss2(aux_pred[task], gt[task], task, self.seg_target, sem_zero)
            out['S4T_'+task+'_pretrain']  = out_loss2(pred_joint[task], gt[task], task, self.seg_target, sem_zero)
            loss_pre_list.append(out['pre_'+task])
            loss_aux_list.append(out['aux_'+task])
            loss_TTT_list.append(out['S4T_'+task+'_pretrain'])

        out['total_pre'] = torch.sum(torch.stack([out['pre_'+t] for t in tasks]))
        out['total_aux_pre'] = torch.sum(torch.stack([out['aux_'+t] for t in tasks]))* self.loss_weights['TTT_loss']
        out['total_S4T_pretrain'] = torch.sum(torch.stack([out['S4T_'+t+'_pretrain'] for t in tasks]))* self.loss_weights['TTT_loss']

        loss_list = [sum(x) for x in zip(loss_pre_list, loss_aux_list, loss_TTT_list)]
        out['total'] = self.AWL(loss_list)

        if self.aux_scale:
            loss_aux_list_aux, loss_TTT_list_aux = [], []
            for task in self.aux_tasks:
                out['aux_'+task] = out_loss2(aux_pred[task], gt[task], task, self.seg_target, sem_zero)
                out['S4T_'+task+'_pretrain']  = out_loss2(pred_joint[task], gt[task], task, self.seg_target, sem_zero)
                loss_aux_list_aux.append(out['aux_'+task])
                loss_TTT_list_aux.append(out['S4T_'+task+'_pretrain'])

            out['total_aux_pre_aux'] = torch.sum(torch.stack([out['aux_'+t] for t in self.aux_tasks]))* self.loss_weights['TTT_loss']
            out['total_S4T_pretrain_aux'] = torch.sum(torch.stack([out['S4T_'+t+'_pretrain'] for t in self.aux_tasks]))* self.loss_weights['TTT_loss']

            loss_list_aux = [sum(x) for x in zip(loss_aux_list_aux, loss_TTT_list_aux)]
            out['total'] += self.AWL2(loss_list_aux) * self.loss_weights['aux_scale']

        if self.ttt:
            loss_list = []
            for task in tasks:
                out['S4T_'+task+'_ttt'] = out_loss(pred_joint[task], pred_pre[task].detach(), self.loss_ft[self.p.TTT_loss], None, self.seg_target, sem_zero)* self.loss_weights['TTT_loss']
                loss_list.append(out['S4T_'+task+'_ttt'])
            out['total_S4T_ttt'] = self.AWL(loss_list)
            out['total'] = out['total_S4T_ttt']
        return out
