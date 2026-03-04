import os
import torch
import numpy as np
import cv2 as cv
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

import config
from network.avatar import AvatarNet
from utils.net_util import to_cuda
from dataset.dataset_mv_rgb_slrf import MvRgbDataset
from utils.renderer import Renderer, gl_perspective_projection_matrix
import utils.recon_util as recon_util
import utils.net_util as net_util
from utils.nerf_util import get_rays

from utils.eval_utils import eval_images


def load_img_mask(data_dir: str, view_idx: int, pose_idx: int):
    img_path = data_dir + '/images/cam%02d/%08d.jpg' % (view_idx, pose_idx)
    if not os.path.exists(img_path):
        img_path = data_dir + '/images/cam%02d/%08d.png' % (view_idx, pose_idx)

    mask_path = data_dir + '/masks/cam%02d/%08d.jpg' % (view_idx, pose_idx)
    if not os.path.exists(mask_path):
        mask_path = data_dir + '/masks/cam%02d/%08d.png' % (view_idx, pose_idx)

    color_img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    mask_img = cv.imread(mask_path, cv.IMREAD_UNCHANGED)

    # Handle multi-channel mask images
    if mask_img is not None and len(mask_img.shape) == 3:
        if mask_img.shape[2] == 2:
            # Grayscale + alpha: use alpha channel
            mask_img = mask_img[:, :, 1]
        elif mask_img.shape[2] == 4:
            # RGBA: use alpha channel
            mask_img = mask_img[:, :, 3]
        else:
            # RGB: convert to grayscale
            mask_img = cv.cvtColor(mask_img, cv.COLOR_BGR2GRAY)
    
    return color_img, mask_img


def test_geometry(network, items, space = 'live', testing_res = (128, 128, 128)):
        if space == 'live':
            bounds = items['live_bounds'][0]
        else:
            bounds = items['cano_bounds'][0]
        vol_pts = net_util.generate_volume_points(bounds, testing_res)
        chunk_size = 256 * 256 * 4
        sdf_list = []
        for i in range(0, vol_pts.shape[0], chunk_size):
            vol_pts_chunk = vol_pts[i: i + chunk_size][None]
            if space == 'live':
                cano_pts_chunk, near_flag = network.transform_live2cano(vol_pts_chunk, items, near_thres = 0.1)
            else:
                cano_pts_chunk = vol_pts_chunk
                near_flag = torch.ones(cano_pts_chunk.shape[:2], dtype = torch.bool)
            sdf_chunk = torch.zeros(cano_pts_chunk.shape[1]).to(cano_pts_chunk)
            if near_flag.sum() > 0:
                ret = network.forward_cano_radiance_field(cano_pts_chunk[near_flag][None], None, items['pose'])
                sdf_chunk[near_flag[0]] = ret['sdf'][0, :, 0]
            sdf_list.append(sdf_chunk)
        sdf_list = torch.cat(sdf_list, 0)
        vertices, faces, normals = recon_util.recon_mesh(sdf_list, testing_res, bounds, iso_value = 0.)
        return vertices, faces, normals


@torch.inference_mode()
def test(test_run, visualize):
    # Load config
    subject_name = test_run['subject_name']
    ckpt_path = test_run['ckpt_path']
    data_path = test_run['data_path']
    start_frame = test_run['start_frame']
    end_frame = test_run['end_frame']
    views = test_run['views']
    device = "cuda"

    # Eval path
    eval_name = f'eval_{subject_name}_frames{start_frame}_{end_frame}_views{"_".join(map(str, views))}.txt'
    eval_path = os.path.join(data_path, eval_name)

    # Adjust global config
    config.opt["train"]["data"] = {
        "data_dir": data_path,
    }

    # Load net
    network = AvatarNet(config.opt['model']).to(device)
    network.eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    network.load_state_dict(ckpt['network'])
    print(f'Loaded checkpoint: {ckpt_path}')

    # Init dataset
    mv_dataset = MvRgbDataset(
        data_dir=data_path,
        frame_range=[start_frame, end_frame, 1],
        used_cam_ids=views,
        subject_name=subject_name,
        training=False
    )
    print(f'Initialized dataset with {len(mv_dataset)} frames.')

    # Clear eval file if it exists
    if os.path.exists(eval_path):
        open(eval_path, 'w').close()

    # Render frames for each view
    all_metrics = []
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for cam_id in views:
        # Get camera parameters
        intr = mv_dataset.intr_mats[cam_id].copy()
        extr = mv_dataset.extr_mats[cam_id].copy()
        img_w = mv_dataset.img_widths[cam_id]
        img_h = mv_dataset.img_heights[cam_id]

        # Init pos renderer
        pos_renderer = Renderer(img_w, img_h, shader_name="position")

        # Eval batching
        batch_size = 1
        ref_imgs = torch.zeros((batch_size, img_h, img_w, 3), dtype=torch.float32).to(device)
        pred_imgs = torch.zeros((batch_size, img_h, img_w, 3), dtype=torch.float32).to(device)

        # Render frames
        for frame_idx in tqdm(range(start_frame, end_frame), desc=f'Rendering cam {cam_id}'):
            # Load reference image and mask
            ref_color_img, mask_img = load_img_mask(data_path, cam_id, frame_idx)
            if ref_color_img is None or mask_img is None:
                ref_imgs[frame_idx % batch_size] = 0
                pred_imgs[frame_idx % batch_size] = 0
                print(f'Warning: Missing image or mask for cam {cam_id}, frame {frame_idx}. Skipping.')
                continue

            # Load test data and move to device
            item = mv_dataset.getitem(
                frame_idx, 
                training=False, 
                extr=extr, 
                intr=intr, 
                img_w=img_w, 
                img_h=img_h
            )
            items = to_cuda(item, add_batch=True)

            # Depth guided sampling
            vertices, faces, normals = test_geometry(
                network, 
                items, 
                "live", 
                testing_res=config.opt['test']['vol_res']
            )
            proj_mat = gl_perspective_projection_matrix(
                intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2], item['img_w'], item['img_h'])
            pos_renderer.set_mvp_mat(proj_mat @ extr)
            pos_renderer.set_model(vertices[faces.reshape(-1)].astype(np.float32))
            pos_map = pos_renderer.render()[..., :3]
            nonzero_flag = np.linalg.norm(pos_map, axis = -1) > 1e-6
            pos_map[nonzero_flag] = np.einsum(
                'ij,vj->vi', extr[:3, :3], pos_map[nonzero_flag]) + extr[:3, 3]
            dist_map = np.linalg.norm(pos_map, axis = -1)

            infer_mask = cv.dilate(nonzero_flag.astype(np.uint8), np.ones((5, 5), np.uint8))
            uv = np.argwhere(infer_mask > 0)[:, [1, 0]].astype(np.int64)
            near = np.zeros(uv.shape[0], np.float32)
            far = np.zeros(uv.shape[0], np.float32)
            ray_d, ray_o = get_rays(uv, item['extr'], item['intr'])
            dist = dist_map[uv[:, 1], uv[:, 0]]

            items['uv'] = torch.from_numpy(uv).to(torch.long).to(device).unsqueeze(0)
            items['near'] = torch.from_numpy(near).to(torch.float32).to(device).unsqueeze(0)
            items['far'] = torch.from_numpy(far).to(torch.float32).to(device).unsqueeze(0)
            items['ray_o'] = torch.from_numpy(ray_o).to(torch.float32).to(device).unsqueeze(0)
            items['ray_d'] = torch.from_numpy(ray_d).to(torch.float32).to(device).unsqueeze(0)
            items['dist'] = torch.from_numpy(dist).to(torch.float32).to(device).unsqueeze(0)

            # Render
            output = network.render(
                items, 
                depth_guided_sampling=config.opt["test"]["depth_guided_sampling"]
            )
            rgb_map = torch.zeros(
                (item['img_h'], item['img_w'], 3), 
                dtype=torch.float32,
                device=device
            ).fill_(0)
            rgb_map[uv[:, 1], uv[:, 0]] = output['rgb_map'][0]
            rgb_map.clip_(0., 1.)
            rgb_map_255 = (rgb_map * 255).to(torch.uint8)

            # Build ground truth image (H, W, 3) in [0, 1], masked
            gt_img = torch.from_numpy(ref_color_img.astype(np.float32) / 255.0).to(device)  # (H, W, 3)
            gt_mask = torch.from_numpy((mask_img > 0).astype(np.float32)).to(device).unsqueeze(-1)  # (H, W, 1)

            gt_img = gt_img * gt_mask  # mask out background
            batch_idx = frame_idx % batch_size
            ref_imgs[batch_idx] = gt_img
            pred_imgs[batch_idx] = rgb_map
            if visualize:
                # cv.imshow('Ground Truth', (gt_img.cpu().numpy() * 255).astype(np.uint8))
                # cv.waitKey(1)
                pass

            # Visualize
            if visualize:
                # cv.imshow('Rendered', rgb_map_255.detach().cpu().numpy())
                # cv.waitKey(1)
                pass

            # Compute metrics for the batch
            if (batch_idx + 1) % batch_size == 0 or frame_idx == end_frame - 1:
                real_batch_size = batch_idx + 1 if frame_idx == end_frame - 1 else batch_size
                # eval_images expects (B, H, W, 3)
                batch_metrics = eval_images(
                    pred_imgs[:real_batch_size], 
                    ref_imgs[:real_batch_size],
                    fid
                )
                all_metrics.append(batch_metrics)
                with open(eval_path, 'a') as f:
                    f.write(f'cam {cam_id} frame {frame_idx - real_batch_size + 1}-{frame_idx}: {batch_metrics}\n')
                # print(f'  cam {cam_id} frame {frame_idx}: {batch_metrics}')

                # Clear cache
                torch.cuda.empty_cache()

    # Average metrics across all frames
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    print(f'Average metrics across all frames and views:')
    for key, value in avg_metrics.items():
        print(f'  {key}: {value}')

    fid_score = fid.compute().item()
    print(f'Final FID score across all frames and views: {fid_score}')

    with open(eval_path, 'a') as f:
        f.write(f'Average metrics across all frames and views: {avg_metrics}\n')
        f.write(f'Final FID score across all frames and views: {fid_score}\n')


tests = [
    # # subject00_julian
    # {
    #     "subject_name": "subject00_julian",
    #     "ckpt_path": "./results/subject00_julian/epoch_latest/net.pt",
    #     "data_path": "./thuman/subject00",
    #     "start_frame": 2000,
    #     "end_frame": 2500,
    #     "views": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    # },
    # {
    #     "subject_name": "subject00_julian",
    #     "ckpt_path": "./results/subject00_julian/epoch_latest/net.pt",
    #     "data_path": "./thuman/subject00",
    #     "start_frame": 0,
    #     "end_frame": 2000,
    #     "views": [23],
    # },
    # # 0165_08
    # {
    #     "subject_name": "0165_08",
    #     "ckpt_path": "./results/0165_08/epoch_latest/net.pt",
    #     "data_path": "./dnarendering/0165_08",
    #     "start_frame": 180,
    #     "end_frame": 225,
    #     "views": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    # },
    {
        "subject_name": "0165_08",
        "ckpt_path": "./results/0165_08/epoch_latest/net.pt",
        "data_path": "./dnarendering/0165_08",
        "start_frame": 0,
        "end_frame": 180,
        "views": [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    }
]


if __name__ == '__main__':
    config.opt = opt = {
        # 'mode': 'train',
        'train': {
            # 'data': {
            #     'subject_name': 'subject00_julian',
            #     'data_dir': './thuman/subject00',
            #     'frame_range': [0, 2000, 1],
            #     'used_cam_ids': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],
            # },
            # 'net_ckpt_dir': './results/subject00_julian',
            # 'prev_ckpt': './results/subject00_julian/epoch_latest',
            # 'save_init_ckpt': False,
            # 'start_epoch': 6,
            # 'end_epoch': 50,
            # 'ckpt_interval': {
            #     'epoch': 1,
            #     'batch': 10000,
            # },
            # 'eval_interval': 10000,
            'depth_guided_sampling': {
                'flag': True,
                'near_sur_dist': 0.05,
                'N_ray_samples': 32,
            },
            'compute_grad': True,
            'ray_sampling': {
                'epoch_ranges': [[0, 30], [30, 9999]],
                'schedules': ['random', 'patch'],
                'type': None,
                'patch': {
                    'patch_num': 1,
                    'patch_size': 64,
                    'inside_radio': 1.0,
                },
                'random': {
                    'sample_num': 1024,
                    'inside_radio': 0.8,
                },
            },
            # 'lr': {
            #     'network': {
            #         'type': 'Step',
            #         'initial': 0.0005,
            #         'interval': 40000,
            #         'factor': 0.9,
            #         'min': 0.00008,
            #     },
            # },
            # 'loss_weight': {
            #     'color': 1.0,
            #     'lpips': 1.0,
            #     'mask': 1.0,
            #     'eikonal': 0.1,
            #     'tv': 10.0,
            # },
            'batch_size': 1,
            'num_workers': 4,
        },
        'test': {
            # 'data': {
            #     'data_dir': './thuman/subject00',
            #     'frame_range': [2000, 2500, 1],
            #     'frame_win': 2,
            #     'fix_head_pose': True,
            # },
            # 'pose_data': {
            #     'data_path': './thuman/subject00/pose_00.npz',
            #     'frame_range': [2000, 2500, 1],
            #     'frame_win': 2,
            # },
            'view_setting': 'free',
            # 'render_view_idx': 23,
            'global_orient': True,
            'img_scale': 1.0,
            'vol_res': [128, 128, 128],
            'depth_guided_sampling': {
                'flag': True,
                'near_sur_dist': 0.02,
                'N_ray_samples': 16,
            },
            'infer_rgb': True,
            'save_mesh': False,
            'render_skeleton': True,
            # 'prev_ckpt': './pretrained_models/subject00_julian',
        },
        'model': {
            'local_pose': True,
            'multires': 6,
            'use_viewdir': False,
            'multires_viewdir': 3,
            'multiscale_line_sizes': [
                [8, 8, 2],
                [32, 32, 8],
                [128, 128, 32],
                [256, 256, 64],
            ],
            'feat_dims': [4, 4, 4, 4],
            'point_nums': [256, 256, 256, 256],
            'knns': [10, 10, 10, 10],
            'pose_formats': ['quaternion', 'quaternion', 'quaternion', 'quaternion'],
            'concat_pose_vec': True,
        },
    }

    for test_run in tests:
        test(test_run, False)
