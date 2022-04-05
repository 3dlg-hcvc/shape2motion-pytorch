import os
import h5py
import logging
import torch

from network import utils
from network.engine import Shape2MotionTrainer

from tools.utils import io
from tools.utils.constant import Stage

log = logging.getLogger('network')

class Network:
    def __init__(self, cfg, input_cfg):

        utils.set_random_seed(cfg.random_seed)
        # torch.set_deterministic(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        self.cfg = cfg
        self.stage = Stage[cfg.name]
        self.input_cfg = input_cfg

        train_path = cfg.train.input_data if io.file_exist(cfg.train.input_data) else input_cfg.train
        test_path = input_cfg.val if cfg.test.split == 'val' else input_cfg.test
        test_path = cfg.test.input_data if io.file_exist(cfg.test.input_data) else test_path
        self.data_path = {"train": train_path, "test": test_path}
        self.trainer = Shape2MotionTrainer(
            cfg=self.cfg,
            data_path=self.data_path,
            stage=self.stage,
        )

    def train(self):
        if not self.cfg.train.continuous:
            if self.stage == Stage.stage2:
                self.trainer.train_with_weights(self.input_cfg.prev_stage_dir)
            else:
                self.trainer.train()
        else:
            self.trainer.resume_train(self.cfg.train.input_model)

    def inference(self):
        self.trainer.test(inference_model=self.cfg.test.inference_model)
