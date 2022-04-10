import os
import logging
import pandas as pd

from omegaconf import OmegaConf

from tools.utils import io
from preprocess.utils import Mat2Hdf5, DatasetName
from multiprocessing import cpu_count

log = logging.getLogger('preprocess')

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
            # config['path'] = os.path.join(self.dataset_dir, train_set)
            # config['set'] = 'train'
            # config['output_path'] = os.path.join(self.output_cfg.path, self.output_cfg.train_data)
            # converter = Mat2Hdf5(config)
            # train_input, train_info = converter.convert(self.split.train.input_file_indices, self.split.train.num_instances)

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

            # input_info = pd.concat([train_input, val_input, test_input], keys=['train', 'val', 'test'], names=['set', 'index'])
            # split_info = pd.concat([train_info, val_info, test_info], keys=['train', 'val', 'test'], names=['set', 'index'])

            # input_info_path = os.path.join(self.tmp_cfg.path, self.tmp_cfg.input_files)
            # input_info.to_csv(input_info_path)

            # split_info_path = os.path.join(self.output_cfg.path, self.output_cfg.split_info)
            # split_info.to_csv(split_info_path)

