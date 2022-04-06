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


log = logging.getLogger('post_stage1')

class PostStage3Impl:
    def __init__(self, cfg):
        self.cfg = cfg
        self.top_k_score_threshold = cfg.top_k_score_threshold

    def rot3d(self, pts, joint_origin, joint_direction):
        joint_direction = joint_direction / LA.norm(joint_direction)
        rot_mat = R.from_rotvec(self.move_angle_param * joint_direction).as_matrix()
        rot_pts = np.dot(pts - joint_origin, rot_mat.transpose()) + joint_origin
        return rot_pts

    def trans3d(self, pts, joint_direction):
        joint_direction = joint_direction / LA.norm(joint_direction)
        shift_vec = joint_direction * self.move_trans_param
        trans_pts = pts + shift_vec
        return trans_pts

    def __call__(self, idx, data):
        data = SimpleNamespace(**data)
        instance_name = data.instance_name
        input_pts = data.input_pts
        input_xyz = input_pts[:, :3]

        pred_anchor_pts = data.pred_anchor_pts
        
        output_data = {}
        output_data['instance_name'] = instance_name
        output_data['input_pts'] = input_pts
        return output_data


class PostStage3:
    def __init__(self, cfg):
        self.cfg = cfg
        self.debug = self.cfg.debug
        self.num_workers = self.cfg.num_workers

        self.gt_h5 = None
        self.df = pd.DataFrame()
        self.output_h5 = None

    def set_datapath(self, data_path, output_path):
        assert io.is_non_zero_file(data_path), OSError(f'Cannot find file {data_path}')
        self.gt_h5 = h5py.File(data_path, 'r')
        io.ensure_dir_exists(os.path.dirname(output_path))
        self.output_h5 = h5py.File(output_path, 'w')

    def process(self, pred, input_pts, gt, id):
        input_pts = input_pts.detach().cpu().numpy()
        
        pred_anchor_pts_batch = pred['anchor_pts'].detach().cpu().numpy()
        pred_joint_direction_cat_batch = pred['joint_direction_cat'].detach().cpu().numpy()
        pred_joint_direction_reg_batch = pred['joint_direction_reg'].detach().cpu().numpy()
        pred_joint_origin_reg_batch = pred['joint_origin_reg'].detach().cpu().numpy()
        pred_joint_type_batch = pred['joint_type'].detach().cpu().numpy()
        pred_simmat_batch = pred['simmat'].detach().cpu().numpy()
        pred_simmat_batch = (pred_simmat_batch <= 255) * pred_simmat_batch + (pred_simmat_batch > 255) * 255
        pred_simmat_batch = pred_simmat_batch.astype(np.uint8)

        batch_size = pred_anchor_pts_batch.shape[0]

        # prepare input data for the stage1 postprocessing
        stage1_data = []
        for b in range(batch_size):
            tmp_data = {}
            instance_name = id[b]
            tmp_data['instance_name'] = instance_name
            tmp_data['pred_anchor_pts'] = pred_anchor_pts_batch[b]
            tmp_data['pred_joint_direction_cat'] = pred_joint_direction_cat_batch[b]

            gt_instance = self.gt_h5[instance_name]
            tmp_data['joint_all_directions'] = gt_instance['joint_all_directions'][:]
            tmp_data['gt_joints'] = gt_instance['gt_joints'][:]
            tmp_data['gt_proposals'] = gt_instance['gt_proposals'][:]

            tmp_data['input_pts'] = input_pts[b]
            stage1_data.append(tmp_data)

        pool = Pool(processes=self.num_workers)
        proc_impl = PostStage3Impl(self.cfg)
        jobs = [pool.apply_async(proc_impl, args=(i,data,)) for i, data in enumerate(stage1_data)]
        pool.close()
        pool.join()
        batch_output = [job.get() for job in jobs]

        for output_data in batch_output:
            if output_data is None:
                continue
            instance_name = output_data['instance_name']
            input_pts = output_data['input_pts']
            motion_scores = output_data['motion_scores']
            
            h5instance = self.output_h5.require_group(instance_name)
            h5instance.create_dataset('input_pts', shape=input_pts.shape, data=input_pts, compression='gzip')
            h5instance.create_dataset('motion_scores', shape=motion_scores.shape, data=motion_scores, compression='gzip')

            if self.debug:
                gt_cfg = {}
    
    def stop(self):
        self.output_h5.close()
