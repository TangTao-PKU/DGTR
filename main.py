import os
import glob
import time
import torch
import random
import logging
import matplotlib
import numpy as np
import os.path as osp
from tqdm import tqdm
from IPython import embed
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from utils.utils import *
from common.utils import *
from utils.eval_utils import *
from model.utils.smpl import SMPL_MODEL_DIR, SMPL
from common.arguments import parse_args, BASE_DATA_DIR
from common.loss import GLoTLoss, CosineAnnealingWarmupRestarts
from common.dataset.data_utils._kp_utils import convert_kps

cfg, cfg_file, args = parse_args()
exec('from model.' + args.model + ' import Model')

def train(cfg, args, train_2d_loader, train_3d_loader, model, optimizer, criterion, train_2d_iter, train_3d_iter):
    losses = AverageMeter()

    model.train()

    for i in tqdm(range(cfg.TRAIN.NUM_ITERS_PER_EPOCH), dynamic_ncols=True):
        # target_2d['kp_2d']: 38, 16, 49, 3     target_3d['kp_3d']: 26, 16, 49, 3
        inp, target_2d, target_3d, train_2d_iter, train_3d_iter = training_data(train_2d_loader, train_3d_loader, train_2d_iter, train_3d_iter)

        preds = model(inp, is_train=True)
        loss, loss_dict = criterion(preds, target_2d, target_3d, None)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

    return losses.avg

def test(model_test, data_test, data_path):
    if args.train:
        model_test.load_state_dict(model.state_dict())

    print(f"Load data from {data_path}")

    model_test.eval()

    J_regressor = torch.from_numpy(np.load(osp.join(BASE_DATA_DIR, 'J_regressor_h36m.npy'))).float()
    full_res = defaultdict(list)

    if 'mpii3d' in data_path:
        J_regressor = None

    for seq_name in tqdm(data_test.keys()):
        chunk_idxes, curr_feat = test_data_1(cfg, data_test, seq_name)

        if chunk_idxes == []:
            continue    

        pred_j3ds, pred_verts, pred_rotmats, pred_thetas = [], [], [], []
        for curr_idx in range(0, len(chunk_idxes), cfg.DATASET.SEQLEN//2):
            # 修改seqlen
            input_feat = test_data_2(cfg,chunk_idxes, curr_idx, curr_feat)

            preds = model_test(input_feat, is_train=False, J_regressor=J_regressor)

            pred_j3ds.append(preds[-1]['kp_3d'].view(-1, preds[-1]['kp_3d'].shape[-2], 3).cpu().numpy())
            pred_verts.append(preds[-1]['verts'].view(-1, 6890, 3).cpu().numpy())
            pred_rotmats.append(preds[-1]['rotmat'].view(-1,24,3,3).cpu().numpy())
            pred_thetas.append(preds[-1]['theta'].view(-1,85).cpu().numpy())

        pred_j3ds = np.vstack(pred_j3ds)
        target_j3ds = data_test[seq_name]['joints3D']
        pred_verts = np.vstack(pred_verts)
        dummy_cam = np.repeat(np.array([[1., 0., 0.]]), len(target_j3ds), axis=0)
        target_theta = np.concatenate([dummy_cam, data_test[seq_name]['pose'], data_test[seq_name]['shape']], axis=1).astype(np.float32)
        target_j3ds, target_theta = target_j3ds[:len(pred_j3ds)], target_theta[:len(pred_j3ds)]

        if 'mpii3d' in data_path:
            
            target_j3ds = convert_kps(target_j3ds, src='spin', dst='mpii3d_test')
            pred_j3ds = convert_kps(pred_j3ds, src='spin', dst='mpii3d_test')

            valid_map = data_test[seq_name]['valid_i'][:,0].nonzero()[0]
            if valid_map.size == 0:
                print("No valid frames. Continue")  # 'subj6_seg0'
                continue

            while True:
                if valid_map[-1] >= len(pred_j3ds):
                    valid_map = valid_map[:-1]
                else:
                    break
        elif target_j3ds.shape[1] == 49:
            target_j3ds = convert_kps(target_j3ds, src='spin', dst='common')
            valid_map = np.arange(len(target_j3ds))
        else:
            valid_map = np.arange(len(target_j3ds))

        pred_j3ds = torch.from_numpy(pred_j3ds).float()
        target_j3ds = torch.from_numpy(target_j3ds).float()

        if 'mpii3d' in data_path:
            pred_pelvis = pred_j3ds[:, [-3], :]
            target_pelvis = target_j3ds[:, [-3], :]
        else:
            pred_pelvis = (pred_j3ds[:, [2], :] + pred_j3ds[:, [3], :]) / 2.0
            target_pelvis = (target_j3ds[:, [2], :] + target_j3ds[:, [3], :]) / 2.0

        pred_j3ds -= pred_pelvis
        target_j3ds -= target_pelvis

        # per-frame accuracy
        mpvpe = compute_error_verts(target_theta=target_theta, pred_verts=pred_verts) * 1000
        mpjpe = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).cpu().numpy()[valid_map]
        mpjpe = mpjpe.mean(axis=-1) * 1000
        S1_hat = compute_similarity_transform(pred_j3ds, target_j3ds)
        mpjpe_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).cpu().numpy()[valid_map]
        mpjpe_pa = mpjpe_pa.mean(axis=-1) * 1000

        accel_err = np.zeros((len(pred_j3ds,)))
        accel_err[1:-1] = compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds) * 1000

        if valid_map[0] == 0:
            valid_map = valid_map[1:]
        if valid_map[-1] == len(accel_err)-1:
            valid_map = valid_map[:-1]
        accel_err = accel_err[valid_map]

        full_res['mpjpe'].append(mpjpe)
        full_res['mpjpe_pa'].append(mpjpe_pa)
        full_res['accel_err'].append(accel_err)
        if args.dataset == '3dpw':
            full_res['mpvpe'].append(mpvpe) 

    full_res.pop(0, None)
    full_res = {k: np.around(np.mean(np.concatenate(v)), 2) for k, v in full_res.items()}

    pa_mpjpe = full_res['mpjpe_pa']
    # print(full_res)

    return pa_mpjpe, full_res


if __name__ == '__main__':
    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    ## model
    model = Model(cfg).cuda()
    
    if args.train:
        # 为每次实验保存模型代码 便于版本管理
        output_model_dir = os.path.join(cfg.LOGDIR, 'trans_t.py')
        shutil.copyfile(src='./model/trans_t.py', dst=output_model_dir)

    model_test = Model(cfg).cuda()
    model_test.regressor.smpl = SMPL(
        SMPL_MODEL_DIR,
        batch_size=64,
        create_transl=False,
        gender='neutral'
    ).cuda()

    ## Load
    if args.checkpoint:
        model_path = sorted(glob.glob(os.path.join(args.checkpoint, '*.pth')))[0]
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint, strict=False)
        model_test.load_state_dict(checkpoint, strict=False)
        print(f"{model_path}")

    optimizer = get_optimizer(model=model, optim_type=cfg.TRAIN.GEN_OPTIM,
        lr=cfg.TRAIN.GEN_LR, weight_decay=cfg.TRAIN.GEN_WD, momentum=cfg.TRAIN.GEN_MOMENTUM)

    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps = cfg.TRAIN.END_EPOCH,
        max_lr=cfg.TRAIN.GEN_LR, min_lr=cfg.TRAIN.GEN_LR * 0.1, warmup_steps=cfg.TRAIN.LR_PATIENCE)

    loss_func = GLoTLoss(
        e_loss_weight=cfg.LOSS.KP_2D_W, e_3d_loss_weight=cfg.LOSS.KP_3D_W,
        e_pose_loss_weight=cfg.LOSS.POSE_W, e_shape_loss_weight=cfg.LOSS.SHAPE_W,
        d_motion_loss_weight=cfg.LOSS.D_MOTION_LOSS_W, use_accel = cfg.LOSS.use_accel, 
        vel_or_accel_2d_weight = cfg.LOSS.vel_or_accel_2d_weight,
        vel_or_accel_3d_weight = cfg.LOSS.vel_or_accel_3d_weight,
    )

    train_2d_loader, train_3d_loader, train_2d_iter, train_3d_iter, dataset_data, data_path, valid_loader = Load_dataset(cfg, args)

    save_name = ''
    best_epoch = 0
    pa_mpjpes = []
    loss_epochs = []
    best_pa_mpjpe = float('inf')

    for epoch in range(1, cfg.TRAIN.END_EPOCH+1):
        if args.train:
            loss = train(cfg, args, train_2d_loader, train_3d_loader, model, optimizer, loss_func, train_2d_iter, train_3d_iter)
            loss_epochs.append(loss)
        
        with torch.no_grad():
            pa_mpjpe, full_res = test(model_test, dataset_data, data_path)
            pa_mpjpes.append(pa_mpjpe)

        if args.train:
            if pa_mpjpe < best_pa_mpjpe:
                best_pa_mpjpe = pa_mpjpe
                best_epoch = epoch
                save_name = save_model(cfg.LOGDIR, save_name, epoch, pa_mpjpe, model)

            logging.info(f'{epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, loss: {loss:.4f}, PA-MPJPE: {pa_mpjpe:.2f}, {best_epoch}_{best_pa_mpjpe:.2f}')
            print(f'{epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, l: {loss:.3f}, PA: {pa_mpjpe:.2f}, {best_epoch}_{best_pa_mpjpe:.2f}')
            lr_scheduler.step()
        else:
            print(full_res)
            break

        start_epoch = 0
        if args.train and epoch > start_epoch:
            plt.figure()
            epoch_x = np.arange(start_epoch+1, len(pa_mpjpes)+1)
            plt.plot(epoch_x, loss_epochs[start_epoch:], '.-', color='C0')
            plt.plot(epoch_x, pa_mpjpes[start_epoch:], '.-', color='C1')
            plt.legend(['Loss', 'Test'])
            plt.ylabel('PAMPJP')
            plt.xlabel('Epoch')
            plt.xlim((start_epoch+1, len(pa_mpjpes)+1))
            plt.savefig(os.path.join(cfg.LOGDIR, 'loss.png'))
            plt.close()

    ## print
    if args.train:
        green = "\033[1;32m%s\033[0m"
        time_now = time.localtime()
        month = int(time.strftime("%m", time_now))
        day = int(time.strftime("%d", time_now))
        hour = int(time.strftime("%H", time_now))
        minute = int(time.strftime("%M", time_now))

        print(green % '\nGPU {}, {}'.format(args.gpu, cfg.LOGDIR))
        print(green % '{}:{} {}:{}, Epoch {}, MPJPE {:.2f}'.format(month, day, hour, minute, best_epoch, best_pa_mpjpe))

        # time.sleep(999999999)
