import os
import logging
from time import time

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from tools.utils import io
from network import Network
from network import utils

from preprocess import PreProcess

log = logging.getLogger('pipeline')

def get_latest_input_cfg(prev_stage_cfg):
    input_cfg = OmegaConf.create()
    prev_stage_dir = os.path.dirname(prev_stage_cfg.path)
    folder, _ = utils.get_latest_file_with_datetime(prev_stage_dir, '', subdir=prev_stage_cfg.inference.folder_name, ext='.h5')
    input_dir = os.path.join(prev_stage_dir, folder, prev_stage_cfg.inference.folder_name)
    input_cfg.train = os.path.join(input_dir, 'train_' + prev_stage_cfg.inference.inference_result)
    input_cfg.val = os.path.join(input_dir, 'val_' + prev_stage_cfg.inference.inference_result)
    input_cfg.test = os.path.join(input_dir, 'test_' + prev_stage_cfg.inference.inference_result)
    input_cfg.prev_stage_dir = prev_stage_dir
    return input_cfg


@hydra.main(config_path='configs', config_name='pipeline')
def main(cfg: DictConfig):
    OmegaConf.update(cfg, "paths.dataset_dir", io.to_abs_path(cfg.paths.dataset_dir, get_original_cwd()))
    OmegaConf.update(cfg, "paths.result_dir", io.to_abs_path(cfg.paths.result_dir, get_original_cwd()))

    if cfg.preprocess.run:
        assert io.folder_exist(cfg.paths.preprocess.input_dir), "Dataset directory doesn't exist"
        io.ensure_dir_exists(cfg.paths.preprocess.output_dir)

        start = time()
        preprocess_stage1 = PreProcess(cfg.preprocess, cfg.paths.preprocess)
        preprocess_stage1.process(cfg.dataset.name)
        end = time()

        duration_time = utils.duration_in_hours(end - start)
        log.info(f'Preprocess: time duration {duration_time}')

    if cfg.network.stage1.run:
        stage1_network = Network(cfg.network.stage1, cfg.paths.preprocess.output)
        if not cfg.network.stage1.eval_only:
            stage1_network.train()
        stage1_network.inference()

    if cfg.network.stage2.run:
        stage2_input_cfg = get_latest_input_cfg(cfg.paths.network.stage1)

        stage2_network = Network(cfg.network.stage2, stage2_input_cfg)
        if not cfg.network.stage2.eval_only:
            stage2_network.train()
        stage2_network.inference()


if __name__ == '__main__':
    start = time()
    main()
    end = time()

    duration_time = utils.duration_in_hours(end - start)
    log.info(f'Pipeline: Total time duration {duration_time}')
