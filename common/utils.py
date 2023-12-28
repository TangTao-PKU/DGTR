import os
import yaml
import time
import torch
import shutil
import joblib
import logging
import operator
import numpy as np
from tqdm import tqdm
from os import path as osp
from functools import reduce
from typing import List, Union
from collections import defaultdict
from collections import OrderedDict
from common.dataset._loaders import get_data_loaders, get_data_loaders_batch
from common.dataset.data_utils._img_utils import split_into_chunks_test
from IPython import embed

def save_model(work_dir, save_name, epoch, pampjpe, model, model_name='model'):
    os.makedirs(work_dir, exist_ok=True)

    if os.path.exists(save_name):
        os.remove(save_name)

    save_name = '%s/%s_%d_%d.pth' % (work_dir, model_name, epoch, pampjpe * 100)

    torch.save(model.state_dict(), save_name)

    return save_name


def get_sequence(start_index, end_index, seqlen=16):
    if end_index - start_index + 1 == seqlen:
        return [i for i in range(start_index, end_index+1)]
    else:
        seq = []
        if start_index == 0:
            for i in range(seqlen - (end_index - start_index + 1)):
                seq.append(start_index)
            for i in range(start_index, end_index + 1):
                seq.append(i)
        else:
            for i in range(start_index, end_index + 1):
                seq.append(i)
            for i in range(seqlen - (end_index - start_index + 1)):
                seq.append(end_index)
        return seq


def Load_dataset(cfg, args):
    data_loaders = get_data_loaders(cfg)

    train_2d_loader, train_3d_loader, valid_loader = data_loaders
    train_2d_iter = train_3d_iter = None

    if train_2d_loader:
        train_2d_iter = iter(train_2d_loader)

    if train_3d_loader:
        train_3d_iter = iter(train_3d_loader)

    ##-----------------------------test-----------------------------##
    set = 'test'
    if args.dataset == '3dpw':
        data_path = f'./data/preprocessed_data/{args.dataset}_{set}_db.pt'  #
    elif args.dataset == 'h36m':
        if cfg.TITLE == 'repr_table4_h36m_mpii3d_model':
            data_path = f'./data/preprocessed_data/{args.dataset}_{set}_25fps_db.pt'  # Table 4
        elif cfg.TITLE == 'repr_table6_3dpw_model':
            data_path = f'./data/preprocessed_data/{args.dataset}_{set}_front_25fps_tight_db.pt'  # Table 6
    elif args.dataset == 'mpii3d':
        set = 'val'
        data_path = f'./data/preprocessed_data/{args.dataset}_{set}_scale12_db.pt'  #

    # print(f"Load data from {data_path}")
    dataset_data = joblib.load(data_path)
    full_res = defaultdict(list)

    vid_name_list = dataset_data['vid_name']
    unique_names = np.unique(vid_name_list)
    data_keyed = {}

    for u_n in unique_names:
        if (args.seq != '') and (not args.seq in u_n):
            continue
        indexes = vid_name_list == u_n
        if 'valid' in dataset_data:
            valids = dataset_data['valid'][indexes].astype(bool)
        else:
            valids = np.ones(dataset_data['features'][indexes].shape[0]).astype(bool)

        data_keyed[u_n] = {
            'features': dataset_data['features'][indexes][valids],
            'joints3D': dataset_data['joints3D'][indexes][valids],
            'vid_name': dataset_data['vid_name'][indexes][valids],
            'imgname': dataset_data['img_name'][indexes][valids],
            'bbox': dataset_data['bbox'][indexes][valids],
        }
        if 'mpii3d' in data_path:
            data_keyed[u_n]['pose'] = np.zeros((len(valids), 72))
            data_keyed[u_n]['shape'] = np.zeros((len(valids), 10))
            data_keyed[u_n]['valid_i'] = dataset_data['valid_i'][indexes][valids]
            J_regressor = None
        else:
            data_keyed[u_n]['pose'] = dataset_data['pose'][indexes][valids]
            data_keyed[u_n]['shape'] = dataset_data['shape'][indexes][valids]
    dataset_data = data_keyed

    return train_2d_loader, train_3d_loader, train_2d_iter, train_3d_iter, dataset_data, data_path, valid_loader

def Load_dataset_batch(cfg, args):
    data_loaders = get_data_loaders_batch(cfg)

    train_2d_loader, train_3d_loader, valid_loader = data_loaders
    train_2d_iter = train_3d_iter = None

    if train_2d_loader:
        train_2d_iter = iter(train_2d_loader)

    if train_3d_loader:
        train_3d_iter = iter(train_3d_loader)

    ##-----------------------------test-----------------------------##
    set = 'test'
    if args.dataset == '3dpw':
        data_path = f'./data/preprocessed_data/{args.dataset}_{set}_db.pt'  #
    elif args.dataset == 'h36m':
        if cfg.TITLE == 'repr_table4_h36m_mpii3d_model':
            data_path = f'./data/preprocessed_data/{args.dataset}_{set}_25fps_db.pt'  # Table 4
        elif cfg.TITLE == 'repr_table6_h36m_model':
            data_path = f'./data/preprocessed_data/{args.dataset}_{set}_front_25fps_tight_db.pt'  # Table 6
    elif args.dataset == 'mpii3d':
        set = 'val'
        data_path = f'./data/preprocessed_data/{args.dataset}_{set}_scale12_db.pt'  #

    # print(f"Load data from {data_path}")
    dataset_data = joblib.load(data_path)
    full_res = defaultdict(list)

    vid_name_list = dataset_data['vid_name']
    unique_names = np.unique(vid_name_list)
    data_keyed = {}

    for u_n in unique_names:
        if (args.seq != '') and (not args.seq in u_n):
            continue
        indexes = vid_name_list == u_n
        if 'valid' in dataset_data:
            valids = dataset_data['valid'][indexes].astype(bool)
        else:
            valids = np.ones(dataset_data['features'][indexes].shape[0]).astype(bool)

        data_keyed[u_n] = {
            'features': dataset_data['features'][indexes][valids],
            'joints3D': dataset_data['joints3D'][indexes][valids],
            'vid_name': dataset_data['vid_name'][indexes][valids],
            'imgname': dataset_data['img_name'][indexes][valids],
            'bbox': dataset_data['bbox'][indexes][valids],
        }
        if 'mpii3d' in data_path:
            data_keyed[u_n]['pose'] = np.zeros((len(valids), 72))
            data_keyed[u_n]['shape'] = np.zeros((len(valids), 10))
            data_keyed[u_n]['valid_i'] = dataset_data['valid_i'][indexes][valids]
            J_regressor = None
        else:
            data_keyed[u_n]['pose'] = dataset_data['pose'][indexes][valids]
            data_keyed[u_n]['shape'] = dataset_data['shape'][indexes][valids]
    dataset_data = data_keyed

    return train_2d_loader, train_3d_loader, train_2d_iter, train_3d_iter, dataset_data, data_path, valid_loader

def move_dict_to_device(dict, device='cuda', tensor2float=False):
    for k,v in dict.items():
        if isinstance(v, torch.Tensor):
            if tensor2float:
                dict[k] = v.float().to(device)
            else:
                dict[k] = v.to(device)


def training_data(train_2d_loader, train_3d_loader, train_2d_iter, train_3d_iter):
    target_2d = target_3d = None
    if train_2d_iter:
        try:
            target_2d = next(train_2d_iter)
        except StopIteration:
            train_2d_iter = iter(train_2d_loader)
            target_2d = next(train_2d_iter)

        move_dict_to_device(target_2d)

    if train_3d_iter:
        try:
            target_3d = next(train_3d_iter)
        except StopIteration:
            train_3d_iter = iter(train_3d_loader)
            target_3d = next(train_3d_iter)

        move_dict_to_device(target_3d)

    if target_2d and target_3d:
        inp = torch.cat((target_2d['features'], target_3d['features']), dim=0).cuda()
    elif target_3d:
        inp = target_3d['features'].cuda()
    else:
        inp = target_2d['features'].cuda()

    return inp, target_2d, target_3d, train_2d_iter, train_3d_iter

def training_data_batch(train_2d_loader, train_3d_loader, train_2d_iter, train_3d_iter):
    target_2d = target_3d = None
    if train_2d_iter:
        try:
            target_2d = next(train_2d_iter)
        except StopIteration:
            train_2d_iter = iter(train_2d_loader)
            target_2d = next(train_2d_iter)

        move_dict_to_device(target_2d)

    if train_3d_iter:
        try:
            target_3d = next(train_3d_iter)
        except StopIteration:
            train_3d_iter = iter(train_3d_loader)
            target_3d = next(train_3d_iter)

        move_dict_to_device(target_3d)

    if target_2d and target_3d:
        # inp = torch.cat((target_2d['features'], target_3d['features']), dim=0).cuda()
        inp = target_3d['features'].cuda()
    elif target_3d:
        inp = target_3d['features'].cuda()
    else:
        inp = target_2d['features'].cuda()

    return inp, target_2d, target_3d, train_2d_iter, train_3d_iter

def test_data_1(cfg, data_test, seq_name):
    curr_feats = data_test[seq_name]['features']
    res_save = {}
    curr_feat = torch.tensor(curr_feats).to('cuda')
    vid_names = data_test[seq_name]['vid_name']

    chunk_idxes = split_into_chunks_test(vid_names, seqlen=cfg.DATASET.SEQLEN, stride=1, is_train=False, match_vibe=False)  # match vibe eval number of poses
    return chunk_idxes, curr_feat


# 修改seqlen
def test_data_2(cfg, chunk_idxes, curr_idx, curr_feat):
    input_feat = []
    if (curr_idx + cfg.DATASET.SEQLEN//2) < len(chunk_idxes):
        for ii in range(cfg.DATASET.SEQLEN//2):
            seq_select = get_sequence(chunk_idxes[curr_idx+ii][0], chunk_idxes[curr_idx+ii][1],seqlen=cfg.DATASET.SEQLEN)
            input_feat.append(curr_feat[None, seq_select, :])
    else:
        for ii in range(curr_idx, len(chunk_idxes)):
            seq_select = get_sequence(chunk_idxes[ii][0], chunk_idxes[ii][1],seqlen=cfg.DATASET.SEQLEN)
            input_feat.append(curr_feat[None, seq_select, :])
    input_feat = torch.cat(input_feat, dim=0)

    return input_feat


    