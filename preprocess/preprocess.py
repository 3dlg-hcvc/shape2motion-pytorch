import os
import logging
import torch
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from dgl.geometry import farthest_point_sampler
from dgl import backend as F

from omegaconf import OmegaConf

from tools.utils import io
from tools.visualizations import Visualizer
from preprocess.utils import Mat2Hdf5, DatasetName
from multiprocessing import cpu_count

log = logging.getLogger('preprocess')

all_direction_kmeans = np.array([
    [1., 0., 0.],
    [-1., 0., 0.],
    [0., 1., 0.],
    [0., -1., 0.],
    [0., 0., 1.],
    [0., 0., -1.],
    [0.57735026, 0.57735026, 0.57735026],
    [0.57735026, 0.57735026, -0.57735026],
    [0.57735026, -0.57735026, 0.57735026],
    [0.57735026, -0.57735026, -0.57735026],
    [-0.57735026, 0.57735026, 0.57735026],
    [-0.57735026, 0.57735026, -0.57735026],
    [-0.57735026, -0.57735026, 0.57735026],
    [-0.57735026, -0.57735026, -0.57735026]
])


def fps(pos, npoints, start_idx=None):
    ctx = F.context(pos)
    B, N, C = pos.shape
    pos = pos.reshape(-1, C)
    dist = F.zeros((B * N), dtype=pos.dtype, ctx=ctx)
    if start_idx is None:
        start_idx = F.randint(shape=(B,), dtype=F.int64,
                              ctx=ctx, low=0, high=N - 1)
    else:
        if start_idx >= N or start_idx < 0:
            raise ValueError("Invalid start_idx, expected 0 <= start_idx < {}, got {}".format(
                N, start_idx))
        start_idx = F.full_1d(B, start_idx, dtype=F.int64, ctx=ctx)
    result = F.zeros((npoints * B), dtype=torch.int64, ctx=ctx)
    farthest_point_sampler(pos, B, npoints, dist, start_idx, result)
    return result.reshape(B, npoints)


class PreProcess:
    def __init__(self, cfg, paths):
        self.cfg = cfg
        self.input_cfg = paths.input
        self.tmp_cfg = paths.tmp_output
        self.output_cfg = paths.output
        self.split = self.cfg.split
        self.debug = self.cfg.debug
        self.dataset_dir = paths.input_dir

        io.ensure_dir_exists(self.output_cfg.path)
        io.ensure_dir_exists(self.tmp_cfg.path)

    def process(self, dataset_name):
        if DatasetName[dataset_name] == DatasetName.SHAPE2MOTION:
            log.info(f'Preprocessing dataset {dataset_name}')
            train_set = self.input_cfg.train_set
            val_set = self.input_cfg.val_set
            test_set = self.input_cfg.test_set

            num_processes = min(cpu_count(), self.cfg.num_workers)

            config = {}
            config['num_processes'] = num_processes
            config['tmp_dir'] = self.tmp_cfg.path
            config['debug'] = self.debug
            config['log'] = log

            log.info(f'Processing Start with {num_processes} workers on train set')
            config['path'] = os.path.join(self.dataset_dir, train_set)
            config['set'] = 'train'
            config['output_path'] = os.path.join(self.output_cfg.path, self.output_cfg.train_data)
            converter = Mat2Hdf5(config)
            train_input, train_info = converter.convert(self.split.train.input_file_indices,
                                                        self.split.train.num_instances)

            log.info(f'Processing Start with {num_processes} workers on val set')
            config['path'] = os.path.join(self.dataset_dir, val_set)
            config['set'] = 'val'
            config['output_path'] = os.path.join(self.output_cfg.path, self.output_cfg.val_data)
            converter = Mat2Hdf5(config)
            val_input, val_info = converter.convert(self.split.val.input_file_indices, self.split.val.num_instances)

            log.info(f'Processing Start with {num_processes} workers on test set')
            config['path'] = os.path.join(self.dataset_dir, test_set)
            config['set'] = 'test'
            config['output_path'] = os.path.join(self.output_cfg.path, self.output_cfg.test_data)
            converter = Mat2Hdf5(config)
            test_input, test_info = converter.convert(self.split.test.input_file_indices, self.split.test.num_instances)

            input_info = pd.concat([train_input, val_input, test_input], keys=['train', 'val', 'test'],
                                   names=['set', 'index'])
            split_info = pd.concat([train_info, val_info, test_info], keys=['train', 'val', 'test'],
                                   names=['set', 'index'])

            input_info_path = os.path.join(self.tmp_cfg.path, self.tmp_cfg.input_files)
            input_info.to_csv(input_info_path)

            split_info_path = os.path.join(self.output_cfg.path, self.output_cfg.split_info)
            split_info.to_csv(split_info_path)
        elif DatasetName[dataset_name] == DatasetName.MULTISCAN:
            log.info(f'Preprocessing dataset {dataset_name}')
            self.process_multiscan_data(os.path.join(self.dataset_dir, self.input_cfg.train_set),
                                        os.path.join(self.output_cfg.path, self.output_cfg.train_data))

            self.process_multiscan_data(os.path.join(self.dataset_dir, self.input_cfg.val_set),
                                        os.path.join(self.output_cfg.path, self.output_cfg.val_data))

            self.process_multiscan_data(os.path.join(self.dataset_dir, self.input_cfg.test_set),
                                        os.path.join(self.output_cfg.path, self.output_cfg.test_data))

    def process_multiscan_data(self, input_file_path, output_file_path):
        num_points = self.cfg.num_points

        h5file = h5py.File(input_file_path, 'r')
        h5output = h5py.File(output_file_path, 'w')
        for key in tqdm(h5file.keys()):
            h5instance = h5file[key]
            num_parts = h5instance.attrs['numParts']
            pts = h5instance['pts'][:]
            part_semantic_masks = h5instance['part_semantic_masks'][:]
            part_instance_masks = h5instance['part_instance_masks'][:]
            joint_types = h5instance['joint_types'][:]
            joint_origins = h5instance['joint_origins'][:]
            joint_axes = h5instance['joint_axes'][:]
            joint_axes = joint_axes / np.linalg.norm(joint_axes, axis=1).reshape(-1, 1)
            joint_ranges = h5instance['joint_ranges'][:]

            if pts.shape[0] == num_points:
                input_pts = pts
            else:
                pcd = torch.from_numpy(pts.reshape((1, pts.shape[0], pts.shape[1])))
                point_idx = fps(pcd, num_points)[0].cpu().numpy()
                input_pts = pts[point_idx, :]
                part_semantic_masks = part_semantic_masks[point_idx]
                part_instance_masks = part_instance_masks[point_idx]

            input_pts[:, 3:6] = input_pts[:, 3:6] / 127.5 - 1.0

            assert num_parts == np.unique(part_instance_masks).shape[0] - 1

            anchor_pts = np.zeros(num_points)
            joint_direction_cat = np.zeros(num_points)
            joint_direction_reg = np.zeros((num_points, 3))
            joint_origin_reg = np.zeros((num_points, 3))
            joint_type = np.zeros(num_points)
            joint_all_directions = all_direction_kmeans
            gt_joints = np.zeros((num_parts, 7))
            gt_proposals = np.zeros((num_parts + 1, num_points))
            simmat = np.zeros((num_points, num_points))

            scale = np.linalg.norm(np.amax(input_pts[:, :3], axis=0) - np.amin(input_pts[:, :3], axis=0))
            input_pts[:, :3] = input_pts[:, :3] / scale

            for i in range(num_parts):
                joint_origin = joint_origins[i, :] / scale
                joint_axis = joint_axes[i, :]

                input_xyz = input_pts[:, :3]
                anchor_disp = input_xyz - joint_origin
                distance = np.linalg.norm(np.cross(anchor_disp, joint_axis), axis=1)
                anchor_indices = np.argsort(distance)[:30]

                rad = np.arccos(np.clip(np.dot(all_direction_kmeans, joint_axis), -1.0, 1.0))
                axis_class = np.argmin(rad)
                axis_delta = joint_axis - all_direction_kmeans[axis_class]
                origin_reg = anchor_disp.dot(joint_axis).reshape(-1, 1) * joint_axis - anchor_disp

                anchor_pts[anchor_indices] = 1
                joint_direction_cat[anchor_indices] = axis_class + 1
                joint_direction_reg[anchor_indices] = axis_delta
                joint_origin_reg[anchor_indices] = origin_reg[anchor_indices]
                joint_type[anchor_indices] = joint_types[i] + 1
                gt_joints[i] = np.concatenate((joint_origin, joint_axis, [joint_types[i] + 1]))
                gt_proposals[i + 1] = part_instance_masks == (i + 1)

            for i in range(num_points):
                simmat[i] = part_instance_masks == part_instance_masks[i]
            np.fill_diagonal(simmat, 0)

            articulation_id = 0
            instance_name = f'{key}_{articulation_id}'
            h5output_inst = h5output.require_group(instance_name)
            h5output_inst.create_dataset('input_pts', shape=input_pts.shape, data=input_pts.astype(np.float32),
                                         compression='gzip')
            h5output_inst.create_dataset('anchor_pts', shape=anchor_pts.shape, data=anchor_pts.astype(np.float32),
                                         compression='gzip')
            h5output_inst.create_dataset('joint_direction_cat', shape=joint_direction_cat.shape,
                                         data=joint_direction_cat.astype(np.float32),
                                         compression='gzip')
            h5output_inst.create_dataset('joint_direction_reg', shape=joint_direction_reg.shape,
                                         data=joint_direction_reg.astype(np.float32),
                                         compression='gzip')
            h5output_inst.create_dataset('joint_origin_reg', shape=joint_origin_reg.shape,
                                         data=joint_origin_reg.astype(np.float32),
                                         compression='gzip')
            h5output_inst.create_dataset('joint_type', shape=joint_type.shape, data=joint_type.astype(np.float32),
                                         compression='gzip')
            h5output_inst.create_dataset('joint_all_directions', shape=joint_all_directions.shape,
                                         data=joint_all_directions.astype(np.float32),
                                         compression='gzip')
            h5output_inst.create_dataset('gt_joints', shape=gt_joints.shape, data=gt_joints.astype(np.float32),
                                         compression='gzip')
            h5output_inst.create_dataset('gt_proposals', shape=gt_proposals.shape, data=gt_proposals.astype(np.float32),
                                         compression='gzip')
            h5output_inst.create_dataset('simmat', shape=simmat.shape, data=simmat.astype(np.float32),
                                         compression='gzip')

            if self.debug:
                viz = Visualizer()
                viz.view_stage1_input(h5output_inst)
        h5output.close()
