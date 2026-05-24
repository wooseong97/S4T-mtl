# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import os
import cv2
import yaml
from easydict import EasyDict as edict
from utils.utils import mkdir_if_missing
import pdb


def parse_task_dictionary(db_name, task_dictionary, val_seg):
    """ 
        Return a dictionary with task information. 
        Additionally we return a dict with key, values to be added to the main dictionary
    """

    task_cfg = edict()
    other_args = dict()
    task_cfg.NAMES = []
    task_cfg.NUM_OUTPUT = {}
    task_cfg.FLAGVALS = {'image': cv2.INTER_CUBIC}
    task_cfg.INFER_FLAGVALS = {}

    if 'include_semseg' in task_dictionary.keys() and task_dictionary['include_semseg']:
        tmp = 'semseg'
        task_cfg.NAMES.append('semseg')
        if db_name == 'PASCALContext':
            if val_seg == 'NYUD':
                task_cfg.NUM_OUTPUT[tmp] = 4
            elif val_seg == 'Taskonomy':
                task_cfg.NUM_OUTPUT[tmp] = 6
            else:
                task_cfg.NUM_OUTPUT[tmp] = 21
        elif db_name == 'NYUD':
            if val_seg == 'PASCALContext':
                task_cfg.NUM_OUTPUT[tmp] = 4
            elif val_seg == 'Taskonomy':
                task_cfg.NUM_OUTPUT[tmp] = 7
            else:
                task_cfg.NUM_OUTPUT[tmp] = 40
        else:
            raise NotImplementedError
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST 
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_NEAREST

    if 'include_depth' in task_dictionary.keys() and task_dictionary['include_depth']:
        tmp = 'depth'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR

    if 'include_human_parts' in task_dictionary.keys() and task_dictionary['include_human_parts']:
        # Human Parts Segmentation
        assert(db_name == 'PASCALContext')
        tmp = 'human_parts'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 7
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_NEAREST

    if 'include_sal' in task_dictionary.keys() and task_dictionary['include_sal']:
        # Saliency Estimation
        assert(db_name == 'PASCALContext')
        tmp = 'sal'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 2
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR

    if 'include_normals' in task_dictionary.keys() and task_dictionary['include_normals']:
        # Surface Normals 
        tmp = 'normals'
        assert(db_name in ['PASCALContext', 'NYUD'])
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 3
        task_cfg.FLAGVALS[tmp] = cv2.INTER_CUBIC
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR
    task_cfg.INFER_FLAGVALS['normals'] = cv2.INTER_LINEAR

    if 'include_edge' in task_dictionary.keys() and task_dictionary['include_edge']:
        # Edge Detection
        assert(db_name in ['PASCALContext', 'NYUD'])
        tmp = 'edge'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR
        other_args['edge_w'] = task_dictionary['edge_w']
        other_args['eval_edge'] = False
    task_cfg.INFER_FLAGVALS['edge'] = cv2.INTER_LINEAR
    return task_cfg, other_args


def parse_task_dictionary_taskonomy(task_dictionary, val_seg):
    task_cfg = edict()
    other_args = dict()
    task_cfg.NAMES = []
    task_cfg.NUM_OUTPUT = {}

    type_to_channel = {'depth_euclidean':1, 'depth_zbuffer':1, 'edge_occlusion':1, 'edge_texture':1, 'keypoints2d':1, 'keypoints3d':1, 'normal':3, 'principal_curvature':2,  'reshading':3, 'rgb':3, 'segment_semantic':7, 'segment_unsup2d':1, 'segment_unsup25d':1}
    if val_seg == 'PASCALContext':
        type_to_channel['segment_semantic'] = 6
    elif val_seg == 'NYUD':
        type_to_channel['segment_semantic'] = 7
    else:
        type_to_channel['segment_semantic'] = 16


    for task in task_dictionary:
        task_cfg.NAMES.append(task)
        task_cfg.NUM_OUTPUT[task] = type_to_channel[task]

    return task_cfg


def dataset_resolution(dataset, backbone):
    if dataset == 'NYUD':
        if 'hrnet' in backbone:
            train_scale = (480, 640); test_scale = (480, 640)
        else:
            # train_scale = (425, 560); test_scale = (425, 560)
            train_scale = (448, 576); test_scale = (448, 576)
    elif dataset == 'PASCALContext':
        train_scale = (512,512); test_scale = (512,512)
    elif dataset == 'Taskonomy':
        train_scale = (256, 256); test_scale = (256, 256)
    else:
        raise NotImplementedError
    
    return train_scale, test_scale


def create_config(exp_file, params):
    
    with open(exp_file, 'r') as stream:
        config = yaml.safe_load(stream)

    # Copy all the arguments
    cfg = edict()
    for k, v in config.items():
        cfg[k] = v

    # set root dir
    root_dir = cfg["out_dir"] + cfg['version_name']
    val_seg = cfg['val_seg']

    # Parse the task dictionary separately
    if cfg['train_db_name'] == 'Taskonomy':
        cfg.TASKS = parse_task_dictionary_taskonomy(cfg['TASKS']['NAMES'], val_seg)
        if 'AUX_TASKS' in cfg.keys():
            cfg.TASKS_aux = parse_task_dictionary_taskonomy(cfg['AUX_TASKS']['NAMES'], val_seg)
            cfg.TASKS_all = {'NAMES': [], 'NUM_OUTPUT': {}}
            cfg.TASKS_all['NAMES'].extend(cfg.TASKS.NAMES); cfg.TASKS_all['NAMES'].extend(cfg.TASKS_aux.NAMES)
            cfg.TASKS_all['NUM_OUTPUT'].update(cfg.TASKS.NUM_OUTPUT); cfg.TASKS_all['NUM_OUTPUT'].update(cfg.TASKS_aux.NUM_OUTPUT)
    else:
        cfg.TASKS, extra_args = parse_task_dictionary(cfg['train_db_name'], cfg['task_dictionary'], val_seg)
        if 'AUX_TASKS' in cfg.keys():
            if 'pretrained_ckpt' in cfg.keys():
                cfg.TASKS_aux = parse_task_dictionary_taskonomy(cfg['AUX_TASKS']['NAMES'], val_seg)
                cfg.TASKS_all = {'NAMES': [], 'NUM_OUTPUT': {}}
                cfg.TASKS_all['NAMES'].extend(cfg.TASKS.NAMES); cfg.TASKS_all['NAMES'].extend(cfg.TASKS_aux.NAMES)
                cfg.TASKS_all['NUM_OUTPUT'].update(cfg.TASKS.NUM_OUTPUT); cfg.TASKS_all['NUM_OUTPUT'].update(cfg.TASKS_aux.NUM_OUTPUT)

        for k, v in extra_args.items():
            cfg[k] = v
    
    # Other arguments
    cfg.TRAIN = edict()
    cfg.TEST = edict()
    cfg.TRAIN.SCALE, cfg.TEST.SCALE = dataset_resolution(cfg['train_db_name'], cfg['backbone'])

    # set log dir
    output_dir = root_dir
    cfg['root_dir'] = root_dir
    cfg['output_dir'] = output_dir
    cfg['save_dir'] = os.path.join(output_dir, 'results')
    cfg['checkpoint'] = os.path.join(output_dir, 'checkpoint.pth.tar')
    if params['run_mode'] != 'infer':
        mkdir_if_missing(cfg['output_dir'])
        mkdir_if_missing(cfg['save_dir'])

    from configs.mypath import db_paths, PROJECT_ROOT_DIR
    params['db_paths'] = db_paths
    params['PROJECT_ROOT_DIR'] = PROJECT_ROOT_DIR

    cfg.update(params)

    return cfg
