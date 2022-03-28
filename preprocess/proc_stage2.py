import os
import logging
import pandas as pd
import numpy as np
import h5py
import pdb

from omegaconf import OmegaConf

from tools.utils import io
from multiprocessing import Pool, cpu_count

log = logging.getLogger('proc_stage2')


class ProcStage2Impl:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, idx, instance_name):
        pass

        

class ProcStage2:
    def __init__(self, cfg):
        self.cfg = cfg
        self.network_stage1_cfg = self.cfg.paths.network.stage1
        self.input_cfg = self.cfg.paths.preprocess.stage2.input
        self.tmp_cfg = self.cfg.paths.preprocess.stage2.tmp_output
        self.output_cfg = self.cfg.paths.preprocess.stage2.output
        self.gt_h5 = None
        self.df = pd.DataFrame()
        self.output_h5 = None

    def set_gt_datapath(self, data_path, data_set):
        assert io.is_non_zero_file(data_path), OSError(f'Cannot find file {data_path}')
        self.gt_h5 = h5py.File(data_path)
        if data_set == 'train':
            self.output_h5 = h5py.File(self.output_cfg.train, 'w')
        else:
            self.output_h5 = h5py.File(self.output_cfg.val, 'w')

    def process(self, pred, input_pts, gt, id):
        input_pts = input_pts.detach().cpu().numpy()
        
        pred_anchor_pts = pred['anchor_pts'].detach().cpu().numpy()
        pred_joint_direction_cat = pred['joint_direction_cat'].detach().cpu().numpy()
        pred_joint_direction_reg = pred['joint_direction_reg'].detach().cpu().numpy()
        pred_joint_origin_reg = pred['joint_origin_reg'].detach().cpu().numpy()
        pred_joint_type = pred['joint_type'].detach().cpu().numpy()
        pred_simmat = pred['simmat'].detach().cpu().numpy()
        pred_simmat = (pred_simmat <= 255) * pred_simmat + (pred_simmat > 255) * 255
        pred_simmat = pred_simmat.astype(np.uint8)
        pred_confidence = pred['confidence'].detach().cpu().numpy()

        batch_size = pred_anchor_pts.shape[0]

        for b in range(batch_size):
            instance_name = id[b]
            pred_joint_type[b]

            pdb.set_trace()

    def stop(self):
        self.output_h5.close()

            





    
