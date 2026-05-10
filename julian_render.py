import os
import torch
import numpy as np
import cv2 as cv
from tqdm import tqdm

import config
from network.avatar import AvatarNet
from utils.net_util import to_cuda
from dataset.dataset_mv_rgb_slrf import MvRgbDataset
from utils.renderer import Renderer, gl_perspective_projection_matrix
import utils.recon_util as recon_util
import utils.net_util as net_util
from utils.nerf_util import get_rays


def test_geometry(network, items, space='live', testing_res=(128, 128, 128)):
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
            cano_pts_chunk, near_flag = network.transform_live2cano(vol_pts_chunk, items, near_thres=0.1)
        else:
            cano_pts_chunk = vol_pts_chunk
            near_flag = torch.ones(cano_pts_chunk.shape[:2], dtype=torch.bool)
        sdf_chunk = torch.zeros(cano_pts_chunk.shape[1]).to(cano_pts_chunk)
        if near_flag.sum() > 0:
            ret = network.forward_cano_radiance_field(cano_pts_chunk[near_flag][None], None, items['pose'])
            sdf_chunk[near_flag[0]] = ret['sdf'][0, :, 0]
        sdf_list.append(sdf_chunk)
    sdf_list = torch.cat(sdf_list, 0)
    vertices, faces, normals = recon_util.recon_mesh(sdf_list, testing_res, bounds, iso_value=0.)
    return vertices, faces, normals


@torch.inference_mode()
def render(test_run):
    subject_name = test_run['subject_name']
    ckpt_path = test_run['ckpt_path']
    data_path = test_run['data_path']
    start_frame = test_run['start_frame']
    end_frame = test_run['end_frame']
    views = test_run['views']
    out_dir = test_run.get(
        'out_dir',
        os.path.join('renders', subject_name, f'frames{start_frame}_{end_frame}')
    )
    device = "cuda"

    # Adjust global config
    config.opt["train"]["data"] = {
        "data_dir": data_path,
    }

    # Load network
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
    print(f'Initialized dataset with {len(mv_dataset)} items.')

    for cam_id in views:
        intr = mv_dataset.intr_mats[cam_id].copy()
        extr = mv_dataset.extr_mats[cam_id].copy()
        img_w = mv_dataset.img_widths[cam_id]
        img_h = mv_dataset.img_heights[cam_id]

        pos_renderer = Renderer(img_w, img_h, shader_name="position")

        cam_out_dir = os.path.join(out_dir, 'cam%02d' % cam_id)
        os.makedirs(cam_out_dir, exist_ok=True)

        for frame_idx in tqdm(range(start_frame, end_frame), desc='Rendering cam %d' % cam_id):
            dataset_idx = frame_idx - start_frame

            item = mv_dataset.getitem(
                dataset_idx,
                training=False,
                extr=extr,
                intr=intr,
                img_w=img_w,
                img_h=img_h
            )
            items = to_cuda(item, add_batch=True)

            # Depth-guided sampling: reconstruct geometry
            vertices, faces, normals = test_geometry(
                network,
                items,
                'live',
                testing_res=config.opt['test']['vol_res']
            )
            proj_mat = gl_perspective_projection_matrix(
                intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2], item['img_w'], item['img_h'])
            pos_renderer.set_mvp_mat(proj_mat @ extr)
            pos_renderer.set_model(vertices[faces.reshape(-1)].astype(np.float32))
            pos_map = pos_renderer.render()[..., :3]
            nonzero_flag = np.linalg.norm(pos_map, axis=-1) > 1e-6
            pos_map[nonzero_flag] = np.einsum(
                'ij,vj->vi', extr[:3, :3], pos_map[nonzero_flag]) + extr[:3, 3]
            dist_map = np.linalg.norm(pos_map, axis=-1)

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
                depth_guided_sampling=config.opt['test']['depth_guided_sampling']
            )
            rgb_map = torch.zeros(
                (item['img_h'], item['img_w'], 3),
                dtype=torch.float32,
                device=device
            )
            rgb_map[uv[:, 1], uv[:, 0]] = output['rgb_map'][0]
            rgb_map.clip_(0., 1.)
            rgb_map_np = (rgb_map.cpu().numpy() * 255).astype(np.uint8)

            out_path = os.path.join(cam_out_dir, '%08d.png' % frame_idx)
            cv.imwrite(out_path, rgb_map_np)

            torch.cuda.empty_cache()

    print('Saved renders to %s' % out_dir)


tests = [
    # subject00_julian
    {
        "subject_name": "subject00_julian",
        "ckpt_path": "./results/subject00_julian/epoch_latest/net.pt",
        "data_path": "./thuman/subject00",
        "start_frame": 0,
        "end_frame": 2500,
        "views": list(range(24)),
    },
    {
        "subject_name": "subject01_julian",
        "ckpt_path": "./results/subject01_julian/epoch_latest/net.pt",
        "data_path": "./thuman/subject01",
        "start_frame": 0,
        "end_frame": 2500,
        "views": list(range(24)),
    },
    {
        "subject_name": "subject02_julian",
        "ckpt_path": "./results/subject02_julian/epoch_latest/net.pt",
        "data_path": "./thuman/subject02",
        "start_frame": 0,
        "end_frame": 2500,
        "views": list(range(24)),
    },
    # DNA Rendering
    {
        "subject_name": "0165_08",
        "ckpt_path": "./results/0165_08/epoch_latest/net.pt",
        "data_path": "./dnarendering/0165_08",
        "start_frame": 0,
        "end_frame": 225,
        "views": list(range(60)),
    },
    {
        "subject_name": "0166_04",
        "ckpt_path": "./results/0166_04/epoch_latest/net.pt",
        "data_path": "./dnarendering/0166_04",
        "start_frame": 0,
        "end_frame": 225,
        "views": list(range(60)),
    },
    {
        "subject_name": "0206_04",
        "ckpt_path": "./results/0206_04/epoch_latest/net.pt",
        "data_path": "./dnarendering/0206_04",
        "start_frame": 0,
        "end_frame": 225,
        "views": list(range(60)),
    },
]


if __name__ == '__main__':
    config.opt = {
        'train': {
            'data': {},
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
            'batch_size': 1,
            'num_workers': 4,
        },
        'test': {
            'view_setting': 'free',
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
            'render_skeleton': False,
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

    print('Found %d test(s). Running them sequentially...' % len(tests))
    for test_run in tests:
        render(test_run)
