import os
import sys
import time
import argparse
import os.path as osp
import logging
import shutil
from yacs.config import CfgNode as CN

# CONSTANTS
GLoT_DB_DIR = './data/preprocessed_data'
AMASS_DIR = './data/amass'
INSTA_DIR = './data/insta_variety'
MPII3D_DIR = './data/mpi_inf_3dhp'
THREEDPW_DIR = './data/3dpw'
H36M_DIR = './data/h36m'
PENNACTION_DIR = './data/penn_action'
POSETRACK_DIR = './data/posetrack'
BASE_DATA_DIR = './data/base_data'

# Configuration variables
cfg = CN()
cfg.TITLE = 'default'
cfg.OUTPUT_DIR = 'results'
cfg.EXP_NAME = 'default'
cfg.DEVICE = 'cuda'
cfg.DEBUG = True
cfg.LOGDIR = ''
cfg.NUM_WORKERS = 8
cfg.DEBUG_FREQ = 1000
cfg.SEED_VALUE = -1
cfg.render = False

cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

cfg.TRAIN = CN()
cfg.TRAIN.DATASETS_2D = ['Insta']
cfg.TRAIN.DATASETS_3D = ['MPII3D']
cfg.TRAIN.DATASET_EVAL = 'ThreeDPW'
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.OVERLAP = 0.25
cfg.TRAIN.DATA_2D_RATIO = 0.5
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.END_EPOCH = 5
cfg.TRAIN.PRETRAINED_REGRESSOR = ''
cfg.TRAIN.PRETRAINED = ''
cfg.TRAIN.RESUME = ''
cfg.TRAIN.NUM_ITERS_PER_EPOCH = 1000
cfg.TRAIN.LR_PATIENCE = 5
cfg.TRAIN.val_epoch=5
# <====== generator optimizer
cfg.TRAIN.GEN_OPTIM = 'Adam'
cfg.TRAIN.GEN_LR = 1e-4
cfg.TRAIN.GEN_WD = 1e-4
cfg.TRAIN.GEN_MOMENTUM = 0.9

cfg.DATASET = CN()
cfg.DATASET.SEQLEN = 20
cfg.DATASET.OVERLAP = 0.5

cfg.LOSS = CN()
cfg.LOSS.KP_2D_W = 60.
cfg.LOSS.KP_3D_W = 30.
cfg.LOSS.SHAPE_W = 0.001
cfg.LOSS.POSE_W = 1.0
cfg.LOSS.D_MOTION_LOSS_W = 1.
cfg.LOSS.vel_or_accel_2d_weight = 50.
cfg.LOSS.vel_or_accel_3d_weight = 100.
cfg.LOSS.use_accel = True

def get_cfg_defaults():
    return cfg.clone()

def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    print('python ' + ' '.join(sys.argv))   

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/config.yaml', help='cfg file path')
    parser.add_argument('--gpu', type=str, default='1', help='gpu num')
    parser.add_argument('--model', type=str, default='tcmr_frame', help='gpu num')
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--debug', action='store_true')

    # evaluation options
    parser.add_argument('--dataset', type=str, default='3dpw', help='pick from 3dpw, mpii3d, h36m')
    parser.add_argument('--seq', type=str, default='', help='render target sequence')
    parser.add_argument('--render', action='store_true', help='render meshes on an rgb video')
    parser.add_argument('--render_plain', action='store_true', help='render meshes on plain background')
    parser.add_argument('--filter', action='store_true', help='apply smoothing filter')
    parser.add_argument('--plot', action='store_true', help='plot acceleration plot graph')
    parser.add_argument('--frame', type=int, default=0, help='render frame start idx')

    args = parser.parse_args()

    if not args.test:
        args.train = True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()
    cfg.render = args.render

    ## log
    log_time = time.strftime('%m%d_%H%M_%S')
    cfg_name = os.path.basename(cfg_file).split('.')[0]
    logdir = f'{log_time}_{cfg_name}_{args.model}'
    print(logdir)

    if args.train:
        logdir = osp.join(cfg.OUTPUT_DIR, logdir)
        os.makedirs(logdir, exist_ok=True)
        shutil.copy(src=cfg_file, dst=osp.join(logdir, 'config.yaml'))

        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(logdir, 'train.log'), level=logging.INFO)

    cfg.LOGDIR = logdir

    return cfg, cfg_file, args
