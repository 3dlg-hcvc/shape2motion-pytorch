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
        stage1_network = Network(cfg.network.stage1, cfg.paths.preprocess)
        if not cfg.network.stage1.eval_only:
            stage1_network.train()
        stage1_network.inference()


if __name__ == '__main__':
    start = time()
    main()
    end = time()

    duration_time = utils.duration_in_hours(end - start)
    log.info(f'Pipeline: Total time duration {duration_time}')
