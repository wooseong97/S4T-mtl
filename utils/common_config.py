# Rewritten based on MTI-Net by Hanrong Ye
# Original authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil

from configs.mypath import PROJECT_ROOT_DIR
import os

import copy

import pdb


def get_backbone(p):
    """ Return the backbone """
    if p['backbone'] == 'resnet18':
        from models.cnn.resnet_FPN import resnet18
        backbone = resnet18(pretrained=True, momentum=p.momentum.base)
        backbone_channels = [64, 128, 256, 512]
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0]//4//2**i, p.TRAIN.SCALE[1]//4//2**i] for i in range(4)] 
        
    elif p['backbone'] == 'resnet34':
        from models.cnn.resnet_FPN import resnet34
        backbone = resnet34(pretrained=True, momentum=p.momentum.base)
        backbone_channels = [64, 128, 256, 512]
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0]//4//2**i, p.TRAIN.SCALE[1]//4//2**i] for i in range(4)] 
        
    elif p['backbone'] == 'resnet50':
        from models.cnn.resnet_FPN import resnet50
        backbone = resnet50(pretrained=True, momentum=p.momentum.base)
        backbone_channels = [256, 512, 1024, 2048]
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0]//4//2**i, p.TRAIN.SCALE[1]//4//2**i] for i in range(4)] 
    
    elif p['backbone'] == 'resnet101':
        from models.cnn.resnet_FPN import resnet101
        backbone = resnet101(pretrained=True, momentum=p.momentum.base)
        backbone_channels = [256, 512, 1024, 2048]
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0]//4//2**i, p.TRAIN.SCALE[1]//4//2**i] for i in range(4)] 
        
    elif p['backbone'] == 'resnet152':
        from models.cnn.resnet_FPN import resnet152
        backbone = resnet152(pretrained=True, momentum=p.momentum.base)
        backbone_channels = [256, 512, 1024, 2048]
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0]//4//2**i, p.TRAIN.SCALE[1]//4//2**i] for i in range(4)]

    else:
        raise NotImplementedError

    return backbone, backbone_channels


def get_head(p, backbone_channels, task):
    """ Return the decoder head """

    if p['head'] == 'mlp':
        from models.heads import MLPHead
        if 'TASKS_all' in p.keys():
            return MLPHead(backbone_channels, p.TASKS_all.NUM_OUTPUT[task])
        else:
            return MLPHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])
    elif p['head'] == 'convhead_inter':
        from models.TTT_MT import ConvHead_inter
        if 'momentum' in p.keys(): momentum=p['momentum'][task]
        else: momentum=0.1
        return ConvHead_inter(backbone_channels, p['head_inter_channel'] ,p.TASKS.NUM_OUTPUT[task], momentum=momentum)
    else:
        raise NotImplementedError
    
    
def get_S4T_decoder(p):
    """ Return the joint MAE-style decoder used for the self-supervised task. """
    assert p['S4T_decoder'] == 'Joint_MAE_v4', \
        "This release only supports the Joint_MAE_v4 decoder (got %s)." % p['S4T_decoder']
    assert p['S4T_model_spec']['backbone'] == 'vitT', \
        "This release only supports the vitT decoder backbone (got %s)." % p['S4T_model_spec']['backbone']

    from models.TTT_model.Joint_MAE import mae_vit_tiny_patch16
    Joint_MAE_config = mae_vit_tiny_patch16
    if 'TASKS_all' in p.keys():
        joint_in_channels = p.head_inter_channel*(len(p.TASKS_all.NAMES))
        out_chans = sum([p.TASKS_all.NUM_OUTPUT[t] for t in p.TASKS_all.NAMES])
    else:
        joint_in_channels = p.head_inter_channel*(len(p.TASKS.NAMES))
        out_chans = sum([p.TASKS.NUM_OUTPUT[t] for t in p.TASKS.NAMES])

    Joint_MAE = Joint_MAE_config(img_size=p.S4T_model_spec.TTT_input_res, \
                                 in_chans=joint_in_channels, \
                                 out_chans=out_chans, \
                                 embed_dim=p['S4T_model_spec']['encoder_embed'], \
                                 depth=p['S4T_model_spec']['encoder_depth'], \
                                 num_heads = p['S4T_model_spec']['encoder_num_heads'], \
                                 decoder_embed_dim=p['S4T_model_spec']['decoder_embed'], \
                                 decoder_depth=p['S4T_model_spec']['decoder_depth'], \
                                 decoder_num_heads = p['S4T_model_spec']['decoder_num_heads'])

    return Joint_MAE


def get_model(p):
    """ Return the model """

    backbone, backbone_channels = get_backbone(p)

    if p['model'] == 'CNN_MTL_S4T':
        feat_channels = sum(p.backbone_channels)
        if 'TASKS_all' in p.keys():
            heads = torch.nn.ModuleDict({task: get_head(p, feat_channels, task) for task in p.TASKS_all.NAMES})
        else:
            heads = torch.nn.ModuleDict({task: get_head(p, feat_channels, task) for task in p.TASKS.NAMES})
        S4T_decoder = get_S4T_decoder(p)
        if p['S4T_decoder'] == 'Joint_MAE_v4':
            from models.TTT_MT import S4T_MAE_v4
            if 'TASKS_all' in p.keys():
                aux_heads = torch.nn.ModuleDict({task: get_head(p, p.head_inter_channel, task) for task in p.TASKS_all.NAMES})
            else:
                aux_heads = torch.nn.ModuleDict({task: get_head(p, p.head_inter_channel, task) for task in p.TASKS.NAMES})
            model = S4T_MAE_v4(p, backbone, backbone_channels, heads, S4T_decoder, aux_heads)
        else:
            raise NotImplementedError('This release only supports the Joint_MAE_v4 decoder.')
    else:
        raise NotImplementedError('Unknown model {}'.format(p['model']))
    return model


"""
    Transformations, datasets and dataloaders
"""
def get_transformations(p):
    """ Return transformations for training and evaluationg """
    from data import transforms
    import torchvision

    # Training transformations
    if p['train_db_name'] == 'NYUD' or p['train_db_name'] == 'PASCALContext':
        train_transforms = torchvision.transforms.Compose([ # from ATRC
            # transforms.Resize(size=p.TRAIN.SCALE),                    
            transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            transforms.RandomCrop(size=p.TRAIN.SCALE, cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TRAIN.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])

        # Testing 
        valid_transforms = torchvision.transforms.Compose([
            # transforms.Resize(size=p.TEST.SCALE),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TEST.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        return train_transforms, valid_transforms

    else:
        return None, None


def get_train_dataset(p, transforms=None):
    """ Return the train dataset """

    db_name = p['train_db_name']
    val_seg = p['val_seg']
    print('Preparing train dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(p.db_paths['PASCALContext'], download=False, split=['train'], transform=transforms, retname=True,
                                          do_semseg='semseg' in p.TASKS.NAMES,
                                          do_edge='edge' in p.TASKS.NAMES,
                                          do_normals='normals' in p.TASKS.NAMES,
                                          do_sal='sal' in p.TASKS.NAMES,
                                          do_human_parts='human_parts' in p.TASKS.NAMES,
                                          overfit=False,
                                          val_seg=val_seg)

    elif db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p.db_paths['NYUD_MT'], download=False, split='train', transform=transforms, do_edge='edge' in p.TASKS.NAMES, 
                                    do_semseg='semseg' in p.TASKS.NAMES, 
                                    do_normals='normals' in p.TASKS.NAMES, 
                                    do_depth='depth' in p.TASKS.NAMES, overfit=False,
                                    val_seg=val_seg)
        
        
    elif db_name == 'Taskonomy':
        from data.dataset_taskonomy import TaskonomyDataset
        # img_types = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d', 'normal', 'principal_curvature', 'reshading', 'rgb', 'segment_unsup2d', 'segment_unsup25d']
        if 'TASKS_all' in p.keys():     
            load_types = copy.deepcopy(p.TASKS_all.NAMES)
        else:
            load_types = copy.deepcopy(p.TASKS.NAMES)
        load_types.append('rgb')
        database = TaskonomyDataset(load_types, split='tiny', partition='train', resize_scale=256, crop_size=256, fliplr=True, val_seg=val_seg)

    return database


def get_train_dataloader(p, dataset, sampler):
    """ Return the train dataloader """
    collate = collate_mil
    trainloader = DataLoader(dataset, batch_size=p['trBatch'], drop_last=True,
                             num_workers=p['nworkers'], collate_fn=collate, pin_memory=True, sampler=sampler)
    return trainloader


def get_test_dataset(p, transforms=None):
    """ Return the test dataset """

    db_name = p['val_db_name']
    val_seg = p['val_seg']
    print('Preparing test dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(p.db_paths['PASCALContext'], download=False, split=['val'], transform=transforms, retname=True,
                                      do_semseg='semseg' in p.TASKS.NAMES,
                                      do_edge='edge' in p.TASKS.NAMES,
                                      do_normals='normals' in p.TASKS.NAMES,
                                      do_sal='sal' in p.TASKS.NAMES,
                                      do_human_parts='human_parts' in p.TASKS.NAMES,
                                    overfit=False,
                                    val_seg=val_seg)
    
    elif db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p.db_paths['NYUD_MT'], download=False, split='val', transform=transforms, do_edge='edge' in p.TASKS.NAMES, 
                                    do_semseg='semseg' in p.TASKS.NAMES, 
                                    do_normals='normals' in p.TASKS.NAMES, 
                                    do_depth='depth' in p.TASKS.NAMES,
                                    val_seg=val_seg)
        
    elif db_name == 'Taskonomy':
        from data.dataset_taskonomy import TaskonomyDataset
        # img_types = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d', 'normal', 'principal_curvature', 'reshading', 'rgb', 'segment_unsup2d', 'segment_unsup25d']
        if 'TASKS_all' in p.keys():     
            load_types = copy.deepcopy(p.TASKS_all.NAMES)
        else:
            load_types = copy.deepcopy(p.TASKS.NAMES)
        
        load_types.append('rgb')
        database = TaskonomyDataset(load_types, split='custom', partition='test', resize_scale=256, crop_size=256,val_seg=val_seg)

    else:
        raise NotImplemented("test_db_name: Choose among PASCALContext and NYUD")

    return database


def get_test_dataloader(p, dataset):
    """ Return the validation dataloader """
    collate = collate_mil
    testloader = DataLoader(dataset, batch_size=p['valBatch'], shuffle=False, drop_last=False,
                            num_workers=p['nworkers'], pin_memory=True, collate_fn=collate)
    return testloader


""" 
    Loss functions 
"""
def get_loss(p, task=None):
    """ Return loss function for a specific task """

    if task == 'edge':
        from losses.loss_functions import BalancedBinaryCrossEntropyLoss
        criterion = BalancedBinaryCrossEntropyLoss(pos_weight=p['task_dictionary']['edge_w'], ignore_index=p.ignore_index)

    elif task == 'semseg' or task == 'human_parts':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(ignore_index=p.ignore_index)

    elif task == 'normals':
        from losses.loss_functions import L1Loss
        criterion = L1Loss(normalize=True, ignore_index=p.ignore_index)

    elif task == 'sal':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(balanced=True, ignore_index=p.ignore_index) 

    elif task == 'depth':
        from losses.loss_functions import L1Loss
        criterion = L1Loss()
        
    ############################################
    elif task == 'TTT_L1':
        from losses.loss_functions import L1Loss
        criterion = L1Loss(normalize=False)

    elif task == 'image':
        from losses.loss_functions import L1Loss
        criterion = L1Loss(normalize=False)

    else:
        criterion = None

    return criterion


def get_criterion(p):
    if 'S4T_decoder' in p.keys() and p['S4T_decoder'] == 'Joint_MAE_v4':
        if 'pretrained_ckpt' in p.keys(): ttt=True
        else: ttt=False

        if p['train_db_name']=='Taskonomy':
            from losses.loss_schemes import MTLoss_S4T_taskonomy_v2
            loss_ft = {}
            loss_ft[p.TTT_loss] = get_loss(p, p.TTT_loss)
            loss_ft = torch.nn.ModuleDict(loss_ft)
            loss_weights = p['loss_kwargs']['loss_weights']
            return MTLoss_S4T_taskonomy_v2(p, p.TASKS.NAMES, loss_ft, loss_weights, ttt)
        else:
            from losses.loss_schemes import MTLoss_S4T_v2
            loss_ft = {task: get_loss(p, task) for task in p.TASKS.NAMES}
            loss_ft[p.TTT_loss] = get_loss(p, p.TTT_loss)
            loss_ft = torch.nn.ModuleDict(loss_ft)
            loss_weights = p['loss_kwargs']['loss_weights']
            return MTLoss_S4T_v2(p, p.TASKS.NAMES, loss_ft, loss_weights, ttt)

    else:
        raise NotImplementedError('This release only supports the Joint_MAE_v4 decoder.')


"""
    Optimizers and schedulers
"""
def get_optimizer(p, model, criterion=None):
    """ Return optimizer for a given model and setup """

    if p['train_db_name']=='Taskonomy':
        print('Optimizer for Taskonomy dataset')
        if 'pretrained_ckpt' in p.keys() and 'S4T_model_spec' in p.keys() and 'ttt_lr' in p.S4T_model_spec.keys():
            if 'loss_kwargs' in p.keys() and 'aux_scale' in p.loss_kwargs.loss_weights.keys():
                print("TTT parameter groups + AWL2")
                params = [{'params': get_lr_params(model, p=p, part='joint_dec'), 'lr': float(p['S4T_model_spec']['ttt_lr'])},
                        {'params': criterion.AWL.parameters()},
                        {'params': criterion.AWL2.parameters()}]
            else:
                print("TTT parameter groups")
                params = [{'params': get_lr_params(model, p=p, part='joint_dec'), 'lr': float(p['S4T_model_spec']['ttt_lr'])},
                        {'params': criterion.AWL.parameters()}]
        else:
            if 'loss_kwargs' in p.keys() and'aux_scale' in p.loss_kwargs.loss_weights.keys():
                print("TTT parameter groups + AWL2")
                params = [{'params': model.parameters()},
                      {'params': criterion.AWL.parameters()},
                      {'params': criterion.AWL2.parameters()}]
            else:
                print("TTT parameter groups")
                params = [{'params': model.parameters()},
                        {'params': criterion.AWL.parameters()}]

    elif 'pretrained_ckpt' in p.keys() and 'S4T_model_spec' in p.keys() and 'ttt_lr' in p.S4T_model_spec.keys():
        print("TTT parameter groups")
        params = [{'params': get_lr_params(model, p=p, part='joint_dec'), 'lr': float(p['S4T_model_spec']['ttt_lr'])}]
    else:
        print('Optimizer uses a single parameter group - (Default)')
        params = model.parameters()

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    # get scheduler    
    if p.scheduler == 'poly':
        from utils.train_utils import PolynomialLR
        scheduler = PolynomialLR(optimizer, p.max_iter, gamma=0.9, min_lr=0)

    return scheduler, optimizer



def get_lr_params(model, p=None, part=None, tasks=None):
    def ismember(comp, tasks):
        exists = False
        for task in tasks:
            exists = exists or comp.find(task) > 0
        return exists
    
    if hasattr(model, 'module'):
        model = model.module
    
    if part == 'task_specific':
        b = [model]
    elif part == 'shared':
        b = [model]
    elif part == 'joint_dec':
        b = [model]

    for i in range(len(b)):
        for name, k in b[i].named_parameters():
            if k.requires_grad:
                if part == 'task_specific':
                    if ismember(name, tasks):
                        yield k
                elif part == 'shared':
                    if ismember(name, tasks):
                        continue
                    else:
                        yield k
                elif part == 'joint_dec':
                    if ismember(name, part):
                        yield k


import math
from utils.config import dataset_resolution

def dataset_config_TTT(p):
    if p['train_db_name'] == 'NYUD':
        if p['val_seg'] == 'PASCALContext': pass
        elif p['val_seg'] == 'Taskonomy':
            p.task_key = {'edge.':'edge_occlusion.', 'normals':'normal', 'semseg':'segment_semantic', 'depth.':'depth_zbuffer.'}
            # p.task_key = {'edge.':'edge_occlusion.', 'normals':'normal', 'semseg':'segment_semantic', 'depth.':'depth_euclidean.'}
        else: NotImplementedError
    elif p['train_db_name'] == 'PASCALContext':
        if p['val_seg'] == 'NYUD': pass
        elif p['val_seg'] == 'Taskonomy':
            p.task_key = {'edge.':'edge_occlusion.', 'normals':'normal', 'semseg':'segment_semantic', 'depth.':'depth_zbuffer.'}
        else: NotImplementedError
    elif p['train_db_name'] == 'Taskonomy':
        if p['val_seg'] == 'NYUD':
            p.task_key = {'edge_occlusion': 'edge', 'normal':'normals', 'segment_semantic':'semseg', 'depth_zbuffer':'depth'}
        elif p['val_seg'] == 'PASCALContext':
            p.task_key = {'edge_occlusion':'edge', 'normal':'normals', 'segment_semantic':'semseg', 'depth_zbuffer':'depth'}
        else: NotImplementedError
    else: NotImplementedError



def load_pretrained_ckpt(p, model, ckp_model, ts_batch=False):
    def ismember(comp, tasks):
        for task in tasks:
            if comp.find(task) > 0:
                return True, task
        return False, None
    
    def resize_pos_embed(posemb, posemb_new, s_res, t_res, patch_size, num_tokens=1):
        print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
        ntok_new = posemb_new.shape[1]
        if num_tokens:
            posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
            ntok_new -= num_tokens
        else:
            posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    
        gs_old = [int(ele/patch_size) for ele in list(s_res)]
        gs_new = [int(ele/patch_size) for ele in list(t_res)]
        
        print('Position embedding grid-size from %s to %s', gs_old, gs_new)
        posemb_grid = posemb_grid.reshape(1, gs_old[0], gs_old[1], -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        return posemb
    
    if 'task_key' in p.keys(): task_key = p.task_key
    else: task_key = {}
    
    for name in model.state_dict().keys():
        detected, task = ismember(name, task_key.keys())
        if detected:
            _name = name.replace(task, task_key[task])
            ckp_model[name] = ckp_model.pop(_name)
        elif 'pos_embed' in name and 'backbone' in name:
            ckp_model[name] # torch.Size([1, 257, 192])
            model.state_dict()[name] # torch.Size([1, 1025, 192])
            
            if hasattr(model, 'module'): patch_size = model.module.backbone.patch_size
            else: patch_size = model.backbone.patch_size
            source_res = dataset_resolution(p['val_seg'], p['backbone'])[0]
            target_res = dataset_resolution(p['train_db_name'], p['backbone'])[0]
            
            pose_embed = resize_pos_embed(ckp_model[name], model.state_dict()[name], source_res, target_res, patch_size)
            ckp_model[name] = pose_embed   

    return ckp_model

