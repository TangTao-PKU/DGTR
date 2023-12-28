import os
import os.path as osp
import sys
object_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(object_path)
from common.arguments import BASE_DATA_DIR
from model.utils.smpl import SMPL, SMPL_MODEL_DIR
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import munch
import yaml
import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import glob
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader
import importlib
from utils.renderer import Renderer
from common.dataset._dataset_demo import CropDataset, FeatureDataset
from utils.demo_utils import download_youtube_clip, convert_crop_cam_to_orig_img, prepare_rendering_results, video_to_images, images_to_video
from IPython import embed


MIN_NUM_FRAMES = 25
random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

def main(args, cfgs, model):
    device = torch.device('cuda')

    """ Prepare input video (images) """
       # 图像序列/视频输入
    if args.img_file!='none':
        """ 准备输入图像序列 """
        image_folder = './data/3dpw/imageFiles/'+args.img_file
        num_frames = len(os.listdir(image_folder))
        # img_shape = cv2.imread(osp.join(image_folder, '000900.jpg')).shape
        img_shape = cv2.imread(osp.join(image_folder, 'image_00000.jpg')).shape
        # img_shape = cv2.imread(osp.join(image_folder, '00000057.jpg')).shape
    else:
        """ 准备输入视频 """
        video_file = args.vid_file
        if video_file.startswith('https://www.youtube.com'):
            print(f"正在下载网络视频 \'{video_file}\'")
            video_file = download_youtube_clip(video_file, './output/demo_output')
            if video_file is None:
                exit('url不存在')
            print(f"视频已保存在 {video_file}...")
        if not os.path.isfile(video_file):
            exit(f"输入视频 \'{video_file}\' 不存在!")
        image_folder, num_frames, img_shape = video_to_images(video_file)
        
    print(f"输入视频总帧数为:{num_frames}\n")
    orig_height, orig_width = img_shape[:2]
    
    output_path = osp.join('./output/'+args.img_file)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    if(os.path.isdir(output_path)):
        shutil.rmtree(output_path)
    # video_file = args.vid_file
    # vid_name = os.path.basename(video_file)
    # output_path = osp.join('./demo_output', os.path.basename(video_file).replace('.mp4', ''))
    # Path(output_path).mkdir(parents=True, exist_ok=True)
    # image_folder, num_frames, img_shape = video_to_images(video_file, img_folder='./demo_output/image', return_info=True)

    # print(f"Input video number of frames {num_frames}\n")
    # orig_height, orig_width = img_shape[:2]

    """ Run tracking """
    total_time = time.time()
    bbox_scale = 1.2
    # run multi object tracker
    mot = MPT(
        device=device,
        batch_size=args.tracker_batch_size,
        display=args.display,
        detector_type=args.detector,
        output_format='dict',
        yolo_img_size=args.yolo_img_size,
    )
    tracking_results = mot(image_folder)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    """ Get GLoT model """
    seq_len = 16

    # Load pretrained weights
    model_path = sorted(glob.glob(os.path.join(args.checkpoint, '*.pth')))[0]
    print(model_path)
    checkpoint = torch.load(model_path)
    # checkpoint = checkpoint['gen_state_dict']
    model.load_state_dict(checkpoint, strict=True)

    # Change mesh gender
    gender = args.gender  # 'neutral', 'male', 'female'
    model.regressor.smpl = SMPL(
        SMPL_MODEL_DIR,
        batch_size=64,
        create_transl=False,
        gender=gender
    ).cuda()

    model.eval()

    # Get feature_extractor
    from model.utils.hmr import hmr
    hmr = hmr().to(device)
    checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
    hmr.load_state_dict(checkpoint['model'], strict=False)
    hmr.eval()

    """ Run GLoT on each person """
    print("\nRunning GLoT on each person tracklet...")
    GLoT_time = time.time()
    GLoT_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None
        bboxes = tracking_results[person_id]['bbox']
        frames = tracking_results[person_id]['frames']

        # Prepare static image features
        dataset = CropDataset(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        crop_dataloader = DataLoader(dataset, batch_size=256, num_workers=16)

        with torch.no_grad():
            feature_list = []
            for i, batch in enumerate(crop_dataloader):
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.to(device)
                feature = hmr.feature_extractor(batch.reshape(-1,3,224,224))
                feature_list.append(feature.cpu())

            del batch

            feature_list = torch.cat(feature_list, dim=0)

        # Encode temporal features and estimate 3D human mesh
        dataset = FeatureDataset(
            image_folder=image_folder,
            frames=frames,
            seq_len=seq_len,
        )
        dataset.feature_list = feature_list

        dataloader = DataLoader(dataset, batch_size=64, num_workers=32)
        with torch.no_grad():
            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

            for i, batch in enumerate(dataloader):
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.to(device)
                # output = model(batch)[0][-1]
                output = model(batch)[-1]

                pred_cam.append(output['theta'][:, :3])
                pred_verts.append(output['verts'])
                pred_pose.append(output['theta'][:, 3:75])
                pred_betas.append(output['theta'][:, 75:])
                pred_joints3d.append(output['kp_3d'])

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)

            del batch

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()

        bboxes[:, 2:] = bboxes[:, 2:] * 1.2
        if args.render_plain:
            pred_cam[:,0], pred_cam[:,1:] = 1, 0  # np.array([[1, 0, 0]])

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        GLoT_results[person_id] = output_dict

    del model
    end = time.time()
    fps = num_frames / (end - GLoT_time)
    print(f'GLoT FPS: {fps:.2f}')
    total_time = time.time() - total_time
    print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
    print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

    if args.save_pkl:
        print(f"Saving output results to \'{os.path.join(output_path, 'GLoT_output.pkl')}\'.")
        joblib.dump(GLoT_results, os.path.join(output_path, "GLoT_output.pkl"))

    """ Render results as a single video """
    renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=args.wireframe)

    output_img_folder = f'{output_path}/_output'
    input_img_folder = f'{output_path}/_input'
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(input_img_folder, exist_ok=True)

    print(f"\nRendering output video, writing frames to {output_img_folder}")
    # prepare results for rendering
    frame_results = prepare_rendering_results(GLoT_results, num_frames)
    mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in GLoT_results.keys()}

    image_file_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])

    for frame_idx in tqdm(range(len(image_file_names))):
        img_fname = image_file_names[frame_idx]
        img = cv2.imread(img_fname)
        input_img = img.copy()
        if args.render_plain:
            img[:] = 0

        if args.sideview:
            side_img = np.zeros_like(img)

        for person_id, person_data in frame_results[frame_idx].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']

            mesh_filename = None
            if args.save_obj:
                mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
                Path(mesh_folder).mkdir(parents=True, exist_ok=True)
                mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

            mc = mesh_color[person_id]

            img = renderer.render(
                img,
                frame_verts,
                cam=frame_cam,
                color=mc,
                mesh_filename=mesh_filename,
            )
            if args.sideview:
                side_img = renderer.render(
                    side_img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    angle=270,
                    axis=[0,1,0],
                )
        if args.sideview:
            img = np.concatenate([img, side_img], axis=1)

        # save output frames
        cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.jpg'), img)
        cv2.imwrite(os.path.join(input_img_folder, f'{frame_idx:06d}.jpg'), input_img)

        if args.display:
            cv2.imshow('Video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if args.display:
        cv2.destroyAllWindows()

    """ Save rendered video """
    # save_output_name = f'{vid_name.replace(".mp4", "")}_output.mp4'
    save_output_name = f'{args.img_file}_output.mp4'
    save_output_path = os.path.join(output_path, save_output_name)
    # save_input_name = f'{vid_name.replace(".mp4", "")}_input.mp4'
    save_input_name = f'{args.img_file}_input.mp4'
    save_input_path = os.path.join(output_path, save_input_name)

    images_to_video(img_folder=output_img_folder, output_vid_file=save_output_path)
    images_to_video(img_folder=input_img_folder, output_vid_file=save_input_path)
    print(f"Saving result video to {os.path.abspath(save_output_path)}")
    # shutil.rmtree(output_img_folder)
    # shutil.rmtree(input_img_folder)
    # if os.path.isfile(video_file):
    #     shutil.rmtree(image_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str, default='sample_video.mp4', help='input video path or youtube link')

    parser.add_argument('--img_file', type=str, default='none', help='input video path or youtube link')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')
    parser.add_argument('--cfgs', help='path to config file', required=True)

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--save_pkl', action='store_true',
                        help='save results to a pkl file')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    parser.add_argument('--gender', type=str, default='neutral',
                        help='set gender of people from (neutral, male, female)')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--render_plain', action='store_true',
                        help='render meshes on plain background')

    parser.add_argument('--gpu', type=int, default='1', help='gpu num')

    parser.add_argument('--model', type=str, default='tcmr', help='gpu num')

    parser.add_argument('--checkpoint', default='', type=str)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


    config_path = args.cfgs
    cfgs = munch.munchify(yaml.safe_load(open(config_path)))

    exec('from model.' + args.model + ' import Model')
    model = Model(cfgs).cuda()

    main(args, cfgs, model)

# export LD_LIBRARY_PATH=/usr/local/lib/
# 失败情况下检测ffmpeg --libdir将没有的库copy进去也可以


# output = model(batch)[0][-1]  mpsnet tcmr
# #output = model(batch)[-1]    gtmr

# model_path = sorted(glob.glob(os.path.join(args.checkpoint, '*.pth.tar')))[0]
# print(model_path)
# checkpoint = torch.load(model_path)
# checkpoint = checkpoint['gen_state_dict']   mpsnet tcmr