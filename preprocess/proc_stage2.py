import os
import logging
import pandas as pd
import h5py
import pdb

from omegaconf import OmegaConf

from tools.utils import io
from multiprocessing import Pool, cpu_count

log = logging.getLogger('proc_stage2')


class ProcStage2Impl:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stage1_pred_h5 = cfg.stage1_pred_h5
        self.gt_h5 = cfg.gt_h5

    def __call__(self, idx, instance_name):
        stage1_pred = self.stage1_pred_h5[instance_name]
        gt = self.gt_h5[instance_name]

        
        

class ProcStage2:
    def __init__(self, cfg):
        self.cfg = cfg
        self.network_stage1_cfg = self.cfg.paths.network.stage1
        self.input_cfg = self.cfg.paths.preprocess.stage2.input
        self.tmp_cfg = self.cfg.paths.preprocess.stage2.tmp_output
        self.output_cfg = self.cfg.paths.preprocess.stage2.output

    def process(self):
        stage1_pred = self.input_cfg.stage1_pred
        gt = self.input_cfg.gt

        assert io.is_non_zero_file(stage1_pred), OSError('Failed to find stage 1 prediction input')
        assert io.is_non_zero_file(gt), OSError('Failed to find ground truth input')

        stage1_pred_h5 = h5py.File(self.input_cfg.stage1_pred, 'r')
        gt_h5 = h5py.File(self.input_cfg.gt, 'r')

        num_processes = min(cpu_count(), self.cfg.num_workers)

        config = OmegaConf.create()
        config.num_processes = num_processes
        config.stage1_pred_h5 = stage1_pred_h5
        config.gt_h5 = gt_h5

        instance_names = stage1_pred_h5.keys()
        pool = Pool(processes=self.num_processes)
        proc_impl = ProcStage2Impl(self.cfg)
        jobs = [pool.apply_async(proc_impl, args=(i,name,)) for i, name in enumerate(instance_names)]
        pool.close()
        pool.join()
        output_filepath_list = [job.get() for job in jobs]

        stage1_pred_h5.close()
        gt_h5.close()





    
