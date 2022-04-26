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


log = logging.getLogger('post_stage3')


class PostStage3:
    def __init__(self, cfg):
        self.cfg = cfg
        self.debug = self.cfg.debug
        self.num_workers = self.cfg.num_workers

        self.gt_h5 = None
        self.df = pd.DataFrame()
        self.output_h5 = None
        self.data_set = 'train'

    def set_datapath(self, data_path, output_path, data_set):
        assert io.is_non_zero_file(data_path), OSError(f'Cannot find file {data_path}')
        self.gt_h5 = h5py.File(data_path, 'r')
        io.ensure_dir_exists(os.path.dirname(output_path))
        self.output_h5 = h5py.File(output_path, 'w')
        self.data_set = data_set

    def process(self, pred, input_pts, gt, id):
        input_pts = input_pts.detach().cpu().numpy()
        
        pred_part_proposal = pred['part_proposal'].detach().cpu().numpy()
        pred_motion_regression = pred['motion_regression'].detach().cpu().numpy()

        gt_part_proposal = gt['part_proposal'].detach().cpu().numpy()
        gt_moved_pcds = gt['moved_pcds'].detach().cpu().numpy()
    
        batch_size = pred_part_proposal.shape[0]

        # prepare input data for the stage3 postprocessing
        for b in range(batch_size):
            instance_name = id[b]
            tmp_pred_part_proposal = np.argmax(pred_part_proposal[b], axis=0)
            tmp_pred_motion_regression = pred_motion_regression[b]

            # instance_data = self.gt_h5[instance_name]
            # tmp_data['good_motion'] = instance_data['good_motion'][:]
            # tmp_data['orig_pred_part_proposal'] = instance_data['pred_part_proposal'][:]

            tmp_gt_part_proposal = gt_part_proposal[b]
            tmp_input_pts = input_pts[b]
            tmp_gt_moved_pcds = gt_moved_pcds[b]
            if self.debug:
                gt_cfg = {}
                gt_cfg['part_proposal'] = tmp_gt_part_proposal
                gt_cfg = SimpleNamespace(**gt_cfg)

                pred_cfg = {}
                pred_cfg['part_proposal'] = tmp_pred_part_proposal
                pred_cfg = SimpleNamespace(**pred_cfg)

                viz = Visualizer(tmp_input_pts[:, :3])
                viz.view_stage3_output(gt_cfg, pred_cfg)
                for i in range(tmp_gt_moved_pcds.shape[0]):
                    viz = Visualizer(tmp_gt_moved_pcds[i, :, :3])
                    viz.view_stage3_output(gt_cfg, pred_cfg)
                    

        
            h5instance = self.output_h5.require_group(instance_name)
            h5instance.create_dataset('part_proposal', shape=tmp_pred_part_proposal.shape, data=tmp_pred_part_proposal, compression='gzip')
            h5instance.create_dataset('motion_regression', shape=tmp_pred_motion_regression.shape, data=tmp_pred_motion_regression, compression='gzip')
    
    def stop(self):
        self.output_h5.close()
