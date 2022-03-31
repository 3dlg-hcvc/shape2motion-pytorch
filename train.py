import os
import h5py
import logging
import torch
from time import time

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from network import Shape2MotionTrainer, utils
from network.utils import Stage
from tools.utils import io

log = logging.getLogger('train')


@hydra.main(config_path="configs", config_name="network")
def main(cfg: DictConfig):
    OmegaConf.update(cfg, "paths.result_dir", io.to_abs_path(cfg.paths.result_dir, get_original_cwd()))

    stage = Stage[cfg.network.stage]

    utils.set_random_seed(cfg.random_seed)
    # torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    if stage == Stage.stage1:
        train_path = cfg.train.input_data if io.file_exist(cfg.train.input_data) else cfg.paths.preprocess.stage1.output.train
        test_path = cfg.paths.preprocess.stage1.output.val if cfg.test.split == 'val' else cfg.paths.preprocess.stage1.output.test
        test_path = cfg.test.input_data if io.file_exist(cfg.test.input_data) else test_path
        data_path = {"train": train_path, "test": test_path}

    trainer = Shape2MotionTrainer(
        cfg=cfg,
        data_path=data_path,
        stage=stage,
    )
    if not cfg.eval_only:
        log.info(f'Train on {train_path}, validate on {test_path}')
        if not cfg.train.continuous:
            trainer.train()
        else:
            trainer.resume_train(cfg.train.input_model)
        trainer.test()
    else:
        log.info(f'Test on {test_path} with inference model {cfg.test.inference_model}')
        trainer.test(inference_model=cfg.test.inference_model)


if __name__ == "__main__":
    start = time()

    main()

    stop = time()

    duration_time = utils.duration_in_hours(stop - start)
    log.info(f'Total time duration: {duration_time}')
