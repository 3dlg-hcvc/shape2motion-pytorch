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

    def __call__(self, idx, data):
        data = SimpleNamespace(**data)
        
        output_data = {}
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
        pred_anchor_feat_3 = pred['anchor_feat_3'].detach().cpu().numpy()

        if self.debug:
            gt_part_proposal = gt['part_proposal'].detach().cpu().numpy()
            gt_motion_scores = gt['motion_scores'].detach().cpu().numpy()
            gt_anchor_mask = gt['anchor_mask'].detach().cpu().numpy()

        # input_pts
        # part_proposal
        # anchor_mask
        # motion_scores

        batch_size = pred_motion_scores.shape[0]
        for b in range(batch_size):
            tmp_data = {}
            instance_name = id[b]
            tmp_data['instance_name'] = instance_name

            tmp_motion_scores = pred_motion_scores[b]
            tmp_anchor_feat_3 = pred_anchor_feat_3[b].transpose()
            
            h5instance = self.output_h5.require_group(instance_name)
            h5instance.create_dataset('motion_scores', shape=tmp_motion_scores.shape, data=tmp_motion_scores, compression='gzip')
            h5instance.create_dataset('anchor_feat_3', shape=tmp_anchor_feat_3.shape, data=tmp_anchor_feat_3, compression='gzip')

            if self.debug:
                gt_cfg = {}
                gt_cfg['part_proposal'] = gt_part_proposal[b]
                gt_cfg['motion_scores'] = gt_motion_scores[b]
                gt_cfg['anchor_mask'] = gt_anchor_mask[b]
                gt_cfg = SimpleNamespace(**gt_cfg)

                pred_cfg = {}
                pred_cfg['motion_scores'] = tmp_motion_scores.flatten()
                pred_cfg = SimpleNamespace(**pred_cfg)

                viz = Visualizer(input_pts[b][:, :3])
                viz.view_stage2_output(gt_cfg, pred_cfg)
    
    def stop(self):
        self.output_h5.close()
