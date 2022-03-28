import os
import logging
import pandas as pd
import pdb

from omegaconf import OmegaConf

from tools.utils import io
from utils import Mat2Hdf5, DatasetName
from multiprocessing import cpu_count

log = logging.getLogger('proc_stage1')

class ProcStage1:
    def __init__(self, cfg):
        self.cfg = cfg
        self.input_cfg = self.cfg.paths.preprocess.stage1.input
        self.tmp_cfg = self.cfg.paths.preprocess.stage1.tmp_output
        self.output_cfg = self.cfg.paths.preprocess.stage1.output
        self.debug = self.cfg.debug

        io.ensure_dir_exists(self.output_cfg.path)
        io.ensure_dir_exists(self.tmp_cfg.path)

    def process(self):
        if DatasetName[self.cfg.dataset.name] == DatasetName.SHAPE2MOTION:
            dataset_dir = self.cfg.paths.preprocess.input_dir
            train_set = self.input_cfg.train_set
            val_set = self.input_cfg.val_set
            test_set = self.input_cfg.test_set

            num_processes = min(cpu_count(), self.cfg.num_workers)

            config = OmegaConf.create()
            config.num_processes = num_processes
            config.tmp_dir = self.tmp_cfg.path

            log.info(f'Stage1 Processing Start with {num_processes} workers on train set')
            config.path = os.path.join(dataset_dir, train_set)
            config.set = 'train'
            config.output_path = os.path.join(self.output_cfg.path, self.output_cfg.train_data)
            converter = Mat2Hdf5(config)
            train_input, train_info = converter.convert()

            log.info(f'Stage1 Processing Start with {num_processes} workers on val set')
            config.path = os.path.join(dataset_dir, val_set)
            config.set = 'val'
            config.output_path = os.path.join(self.output_cfg.path, self.output_cfg.val_data)
            converter = Mat2Hdf5(config)
            val_input, val_info = converter.convert()

            log.info(f'Stage1 Processing Start with {num_processes} workers on test set')
            config.path = os.path.join(dataset_dir, test_set)
            config.set = 'test'
            config.output_path = os.path.join(self.output_cfg.path, self.output_cfg.test_data)
            converter = Mat2Hdf5(config)
            test_input, test_info = converter.convert()

            input_info = pd.concat([train_input, val_input, test_input], keys=['train', 'val', 'test'], names=['set', 'index'])
            split_info = pd.concat([train_info, val_info, test_info], keys=['train', 'val', 'test'], names=['set', 'index'])

            input_info_path = os.path.join(self.tmp_cfg.path, self.tmp_cfg.input_files)
            input_info.to_csv(input_info_path)

            split_info_path = os.path.join(self.output_cfg.path, self.output_cfg.split_info)
            split_info.to_csv(split_info_path)

