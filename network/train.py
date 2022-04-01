import os
import h5py
import logging
import torch

from network import Shape2MotionTrainer, utils

from tools.utils import io
from tools.utils.constant import Stage
from network import utils

log = logging.getLogger('network')

class Network:
    def __init__(self, cfg, input_cfg):

        utils.set_random_seed(cfg.random_seed)
        # torch.set_deterministic(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        stage = Stage[cfg.name]

        if stage == Stage.stage1:
            train_path = cfg.train.input_data if io.file_exist(cfg.train.input_data) else input_cfg.output.train
            test_path = input_cfg.output.val if cfg.test.split == 'val' else input_cfg.output.test
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
