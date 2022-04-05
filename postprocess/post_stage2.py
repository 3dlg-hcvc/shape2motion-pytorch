import os
import logging
import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R
import h5py
from enum import Enum
from multiprocessing import Pool, cpu_count
from types import SimpleNamespace

from tools.utils import io
from tools.utils.constant import JointType
from tools.visualizations import Visualizer

import pdb

log = logging.getLogger('post_stage2')

class PostStage2Impl:
    def __init__(self, cfg):
        self.cfg = cfg
        self.top_k_score_threshold = cfg.top_k_score_threshold
        self.move_angle_params = cfg.move_angle_params
        self.move_trans_params = cfg.move_trans_params

    def move_pts_with_joint(self, pts, joint_origin, joint_direction, joint_type, angle=np.pi, trans=0.5):
        if joint_type == JointType.ROT.value:
            move_pts = self.rot3d(pts, joint_origin, joint_direction, angle)
        elif joint_type == JointType.TRANS.value:
            move_pts = self.trans3d(pts, joint_direction, trans)
        elif joint_type == JointType.BOTH.value:
            move_pts = self.rot3d(move_pts, joint_origin, joint_direction, angle)
            move_pts = self.trans3d(pts, joint_direction, trans)
        else:
            log.warn(f'No implementation for the joint type value {joint_type}')
        return move_pts

    def rot3d(self, pts, joint_origin, joint_direction, angle):
        joint_direction = joint_direction / LA.norm(joint_direction)
        rot_mat = R.from_rotvec(angle * joint_direction).as_matrix()
        rot_pts = np.dot(pts - joint_origin, rot_mat.transpose()) + joint_origin
        return rot_pts

    def trans3d(self, pts, joint_direction, trans):
        joint_direction = joint_direction / LA.norm(joint_direction)
        shift_vec = joint_direction * trans
        trans_pts = pts + shift_vec
        return trans_pts

    def __call__(self, idx, data):
        data = SimpleNamespace(**data)
        instance_name = data.instance_name
        input_pts = data.input_pts
        num_points = input_pts.shape[0]
        input_xyz = input_pts[:, :3]

        pred_part_proposal = data.pred_part_proposal.astype(bool)
        gt_part_proposal = data.gt_part_proposal
        pred_motions = data.pred_motions
        gt_motion = data.gt_motion
        pred_motion_scores = data.pred_motion_scores.flatten()
        gt_motion_scores = data.gt_motion_scores

        good_motion_idx = pred_motion_scores[::-1].argsort()[:self.top_k_score_threshold]
        # select one good predicted motion
        good_motion_idx = np.random.choice(good_motion_idx.shape[0], 1, replace=False)
        good_motion = pred_motions[good_motion_idx, :].flatten()
        motion_regression = gt_motion - good_motion

        assert len(self.move_angle_params) == len(self.move_trans_params), 'move_angle_params should have the same length as move_trans_params'
        moved_pcds = np.zeros((3, num_points, 6))
        for i in range(len(self.move_angle_params)):
            move_angle = float(self.move_angle_params[i]) / 180.0 * np.pi
            diag_length = LA.norm(np.amax(input_xyz, axis=0) - np.amin(input_xyz, axis=0))
            move_trans = self.move_trans_params[i] * diag_length
            move_pts = self.move_pts_with_joint(input_xyz[pred_part_proposal, :], good_motion[:3], good_motion[3:6], good_motion[-1], move_angle, move_trans)
            tmp_pts = np.copy(input_pts)
            tmp_pts[pred_part_proposal, :3] = move_pts
            moved_pcds[i, :, :] = tmp_pts

        output_data = {}
        output_data['instance_name'] = instance_name
        output_data['input_pts'] = input_pts
        output_data['gt_part_proposal'] = gt_part_proposal
        output_data['motion_regression'] = motion_regression
        output_data['moved_pcds'] = moved_pcds
        return output_data


class PostStage2:
    def __init__(self, cfg):
        self.cfg = cfg
        self.debug = self.cfg.debug
        self.num_workers = self.cfg.num_workers

        self.output_h5 = None

    def set_datapath(self, data_path, output_path):
        assert io.is_non_zero_file(data_path), OSError(f'Cannot find file {data_path}')
        self.gt_h5 = h5py.File(data_path, 'r')
        io.ensure_dir_exists(os.path.dirname(output_path))
        self.output_h5 = h5py.File(output_path, 'w')

    def process(self, pred, input_pts, gt, id):
        input_pts = input_pts.detach().cpu().numpy()

        pred_motion_scores = pred['motion_scores'].detach().cpu().numpy()

        part_proposal = gt['part_proposal'].detach().cpu().numpy()
        gt_motion_scores = gt['motion_scores'].detach().cpu().numpy()
        anchor_mask = gt['anchor_mask'].detach().cpu().numpy()

        batch_size = pred_motion_scores.shape[0]

        stage2_data = []
        for b in range(batch_size):
            tmp_data = {}
            instance_name = id[b]
            tmp_data['instance_name'] = instance_name

            components = instance_name.split('_')
            object_instance_name = '_'.join(components[:-1])
            proposal_idx = int(components[-1])
            object_data = self.gt_h5[object_instance_name]

            tmp_data['input_pts'] = input_pts[b]
            tmp_data['pred_part_proposal'] = part_proposal[b]
            tmp_data['gt_part_proposal'] = object_data['gt_part_proposals'][:][proposal_idx]
            tmp_data['pred_motions'] = object_data['pred_motions'][:]
            tmp_data['gt_motion'] = object_data['gt_motions'][:][proposal_idx]

            tmp_data['pred_motion_scores'] = pred_motion_scores[b]
            tmp_data['gt_motion_scores'] = gt_motion_scores[b]
            stage2_data.append(tmp_data)

            if self.debug:
                gt_cfg = {}
                gt_cfg['part_proposal'] = tmp_data['gt_part_proposal']
                gt_cfg['motions'] = tmp_data['pred_motions']
                gt_cfg['scores'] = tmp_data['gt_motion_scores']
                gt_cfg['anchor_mask'] = anchor_mask[b]
                gt_cfg = SimpleNamespace(**gt_cfg)

                pred_cfg = {}
                pred_cfg['part_proposal'] = tmp_data['pred_part_proposal']
                pred_cfg['motions'] = tmp_data['pred_motions']
                pred_cfg['scores'] = tmp_data['pred_motion_scores']
                pred_cfg['anchor_mask'] = anchor_mask[b]
                pred_cfg = SimpleNamespace(**pred_cfg)

                viz = Visualizer(tmp_data['input_pts'][:, :3])
                viz.view_stage2_output(gt_cfg, pred_cfg)
        
        # pool = Pool(processes=self.num_workers)
        proc_impl = PostStage2Impl(self.cfg)
        # jobs = [pool.apply_async(proc_impl, args=(i,data,)) for i, data in enumerate(stage2_data)]
        # pool.close()
        # pool.join()
        # batch_output = [job.get() for job in jobs]

        batch_output = []
        for i, data in enumerate(stage2_data):
            batch_output.append(proc_impl(i, data))

        for output_data in batch_output:
            if output_data is None:
                continue
            instance_name = output_data['instance_name']
            input_pts = output_data['input_pts']
            part_proposal = output_data['gt_part_proposal']
            motion_regression = output_data['motion_regression']
            moved_pcds = output_data['moved_pcds']
            
            h5instance = self.output_h5.require_group(instance_name)
            h5instance.create_dataset('input_pts', shape=input_pts.shape, data=input_pts, compression='gzip')
            h5instance.create_dataset('part_proposal', shape=part_proposal.shape, data=part_proposal, compression='gzip')
            h5instance.create_dataset('motion_regression', shape=motion_regression.shape, data=motion_regression, compression='gzip')
            h5instance.create_dataset('moved_pcds', shape=moved_pcds.shape, data=moved_pcds, compression='gzip')

    def stop(self):
        self.output_h5.close()
