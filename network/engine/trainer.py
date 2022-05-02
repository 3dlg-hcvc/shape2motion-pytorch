import os
import h5py
import logging
import numpy as np
from time import time
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim import optimizer
from torch.utils.tensorboard import SummaryWriter

from network.model import Shape2Motion
from network.data import Shape2MotionDataset
from network import utils
from network.utils import AvgRecorder
from tools.utils import io
from tools.utils.constant import Stage

from postprocess import PostStage1, PostStage2, PostStage3


class Shape2MotionTrainer:
    def __init__(self, cfg, data_path, stage):
        self.cfg = cfg
        self.log = logging.getLogger('Network')
        # data_path is a dictionary {'train', 'test'}
        if cfg.device == 'cuda:0' and torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        self.device = device
        self.log.info(f'Using device {self.device}')

        self.stage = Stage[stage] if isinstance(stage, str) else stage

        self.train_cfg = self.cfg.paths.train
        self.test_cfg = self.cfg.paths.inference

        self.max_epochs = cfg.train.max_epochs
        self.model = self.build_model()
        self.model.to(device)
        self.log.info(f'Below is the network structure:\n {self.model}')

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.train.lr
        )

        self.data_path = data_path
        self.writer = None

        self.train_loader = None
        self.test_loader = None
        self.init_data_loader(self.cfg.eval_only)

        self.postprocess = None
        if self.stage == Stage.stage1:
            self.postprocess = PostStage1(self.cfg.postprocess)
        elif self.stage == Stage.stage2:
            self.postprocess = PostStage2(self.cfg.postprocess)
        elif self.stage == Stage.stage3:
            self.postprocess = PostStage3(self.cfg.postprocess)

    def build_model(self):
        model = Shape2Motion(self.stage, self.device, self.cfg.num_points, self.cfg.num_channels)
        return model

    def init_data_loader(self, eval_only):
        if not eval_only:
            self.log.info(f'Train on {self.data_path["train"]}, validate on {self.data_path["test"]}')
            self.train_loader = torch.utils.data.DataLoader(
                Shape2MotionDataset(
                    self.data_path['train'], num_points=self.cfg.num_points, stage=self.stage
                ),
                batch_size=self.cfg.train.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                pin_memory=True
            )

            self.log.info(f'Num {len(self.train_loader)} batches in train loader')
        else:
            self.train_loader = torch.utils.data.DataLoader(
                Shape2MotionDataset(
                    self.data_path['train'], num_points=self.cfg.num_points, stage=self.stage
                ),
                batch_size=self.cfg.test.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=True
            )

            self.log.info(f'Num {len(self.train_loader)} batches in train loader')

        self.test_loader = torch.utils.data.DataLoader(
            Shape2MotionDataset(
                self.data_path['test'], num_points=self.cfg.num_points, stage=self.stage
            ),
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )
        self.log.info(f'Num {len(self.test_loader)} batches in test loader')

    def train_epoch(self, epoch):
        self.log.info(f'>>>>>>>>>>>>>>>> Train Epoch {epoch} >>>>>>>>>>>>>>>>')

        self.model.train()

        iter_time = AvgRecorder()
        io_time = AvgRecorder()
        to_gpu_time = AvgRecorder()
        network_time = AvgRecorder()
        start_time = time()
        end_time = time()
        remain_time = ''

        epoch_loss = {
            'total_loss': AvgRecorder()
        }

        for i, (input_pts, gt_dict, id) in enumerate(tqdm(self.train_loader, mininterval=5.0)):
            io_time.update(time() - end_time)
            # Move the tensors to the device
            s_time = time()
            input_pts = input_pts.to(self.device)
            gt = {}
            for k, v in gt_dict.items():
                gt[k] = v.to(self.device)
            to_gpu_time.update(time() - s_time)

            # Get the loss
            s_time = time()
            pred = self.model(input_pts, gt)
            loss_dict = self.model.losses(pred, gt)
            network_time.update(time() - s_time)

            loss = torch.tensor(0.0, device=self.device)
            loss_weight = self.cfg.train.loss_weight
            # use different loss weight to calculate the final loss
            for k, v in loss_dict.items():
                if 'loss' in k:
                    if k not in loss_weight:
                        raise ValueError(f'No loss weight for {k}')
                    loss += loss_weight[k] * v

            # Used to calculate the avg loss
            for k, v in loss_dict.items():
                if k not in epoch_loss.keys():
                    epoch_loss[k] = AvgRecorder()
                epoch_loss[k].update(v)
            epoch_loss['total_loss'].update(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # time and print
            current_iter = epoch * len(self.train_loader) + i + 1
            max_iter = (self.max_epochs + 1) * len(self.train_loader)
            remain_iter = max_iter - current_iter

            iter_time.update(time() - end_time)
            end_time = time()

            remain_time = remain_iter * iter_time.avg
            remain_time = utils.duration_in_hours(remain_time)

        self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
        # self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], epoch)
        # self.scheduler.step()

        # Add the loss values into the tensorboard
        for k, v in epoch_loss.items():
            if k == 'total_loss':
                self.writer.add_scalar(f'{k}', epoch_loss[k].avg, epoch)
            elif 'loss' in k:
                self.writer.add_scalar(f'loss/{k}', epoch_loss[k].avg, epoch)
            else:
                self.writer.add_scalar(f'accuracy/{k}', epoch_loss[k].avg, epoch)

        if epoch % self.cfg.train.log_frequency == 0:
            loss_log = ''
            for k, v in epoch_loss.items():
                loss_log += '{}: {:.5f}  '.format(k, v.avg)

            self.log.info(
                'Epoch: {}/{} Loss: {} io_time: {:.2f}({:.4f}) to_gpu_time: {:.2f}({:.4f}) network_time: {:.2f}({:.4f}) \
                duration: {:.2f} remain_time: {}'
                    .format(epoch, self.max_epochs, loss_log, io_time.sum, io_time.avg, to_gpu_time.sum,
                            to_gpu_time.avg, network_time.sum, network_time.avg, time() - start_time, remain_time))

    def eval_epoch(self, epoch, save_results=False, data_set='test'):
        self.log.info(f'>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        val_loss = {
            'total_loss': AvgRecorder()
        }

        # test the model on the val set and write the results into tensorboard
        self.model.eval()
        data_loader = self.test_loader if data_set == 'test' else self.train_loader
        with torch.no_grad():
            start_time = time()
            for i, (input_pts, gt_dict, id) in enumerate(tqdm(data_loader)):
                # Move the tensors to the device
                input_pts = input_pts.to(self.device)
                gt = {}
                for k, v in gt_dict.items():
                    gt[k] = v.to(self.device)

                pred = self.model(input_pts, gt)
                if save_results:
                    self.postprocess.process(pred, input_pts, gt, id)
                
                loss_dict = self.model.losses(pred, gt)
                loss_weight = self.cfg.train.loss_weight
                loss = torch.tensor(0.0, device=self.device)
                # use different loss weight to calculate the final loss
                for k, v in loss_dict.items():
                    if 'loss' in k:
                        if k not in loss_weight:
                            raise ValueError(f'No loss weight for {k}')
                        loss += loss_weight[k] * v

                # Used to calculate the avg loss
                for k, v in loss_dict.items():
                    if k not in val_loss.keys():
                        val_loss[k] = AvgRecorder()
                    val_loss[k].update(v)
                val_loss['total_loss'].update(loss)
        # write the val_loss into the tensorboard
        if self.writer is not None:
            for k, v in val_loss.items():
                if 'loss' in k:
                    self.writer.add_scalar(f'val_loss/{k}', val_loss[k].avg, epoch)
                else:
                    self.writer.add_scalar(f'val_accuracy/{k}', val_loss[k].avg, epoch)

        loss_log = ''
        for k, v in val_loss.items():
            loss_log += '{}: {:.5f}  '.format(k, v.avg)

        self.log.info(
            'Eval Epoch: {}/{} Loss: {} duration: {:.2f}'
                .format(epoch, self.max_epochs, loss_log, time() - start_time))

        return val_loss

    def train(self, start_epoch=0):
        self.model.train()
        self.writer = SummaryWriter(self.train_cfg.output_dir)

        io.ensure_dir_exists(self.train_cfg.output_dir)

        best_model = None
        best_result = np.inf
        for epoch in range(start_epoch, self.max_epochs + 1):
            self.train_epoch(epoch)

            if epoch % self.cfg.train.save_frequency == 0 or epoch == self.max_epochs:
                # Save the model
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    },
                    os.path.join(self.train_cfg.output_dir,
                                 self.train_cfg.model_filename % epoch),
                )

                val_error = self.eval_epoch(epoch)

                if best_model is None or val_error['total_loss'].avg < best_result:
                    best_model = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }
                    best_result = val_error['total_loss'].avg
                    torch.save(
                        best_model,
                        os.path.join(self.train_cfg.output_dir,
                                     self.train_cfg.best_model_filename)
                    )
        self.writer.close()

    def get_latest_model_path(self, with_best=False, stage_dir=None):
        if stage_dir is None or not io.folder_exist(stage_dir):
            stage_dir = os.path.dirname(self.cfg.paths.path)
        
        folder, filename = utils.get_latest_file_with_datetime(stage_dir, '', subdir=self.train_cfg.folder_name, ext='.pth')
        model_dir = os.path.join(stage_dir, folder, self.train_cfg.folder_name)
        model_path = os.path.join(model_dir, filename)
        if with_best:
            model_path = os.path.join(model_dir, self.train_cfg.best_model_filename)
        return model_path

    def test(self, inference_model=None):
        if not inference_model or not io.file_exist(inference_model):
            self.log.info(f'Loading from the most recently saved model')
            inference_model = self.get_latest_model_path(with_best=self.cfg.test.with_best)
        if not io.file_exist(inference_model):
            raise IOError(f'Cannot open inference model {inference_model}')
        date_dir = os.path.dirname(os.path.dirname(inference_model))
        test_output_dir = os.path.join(date_dir, self.test_cfg.folder_name)
        self.test_cfg.output_dir = test_output_dir
        io.ensure_dir_exists(self.test_cfg.output_dir)

        self.log.info(f'Inference on {self.data_path} with inference model {inference_model}')

        # Load the model
        self.log.info(f'Load model from {inference_model}')
        checkpoint = torch.load(inference_model, map_location=self.device)
        epoch = checkpoint['epoch']
        self.log.info(f'Inferencing with model at epoch {epoch}')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # Save the prediction results into hdf5
        data_sets = ['train', 'test']
        # if self.cfg.eval_only:
        #     data_sets = ['test']
        for data_set in data_sets:
            if data_set == 'train':
                output_path = os.path.join(self.test_cfg.output_dir, f'{data_set}_' + self.test_cfg.inference_result)
            else:
                output_path = os.path.join(self.test_cfg.output_dir, f'{self.cfg.test.split}_' + self.test_cfg.inference_result)
            self.postprocess.set_datapath(self.data_path[data_set], output_path, data_set)
            self.eval_epoch(epoch, save_results=True, data_set=data_set)
            self.postprocess.stop()

    def resume_train(self, model_path=None):
        if not model_path or not io.file_exist(model_path):
            model_path = self.get_latest_model_path()
        # Load the model
        if io.is_non_zero_file(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            epoch = checkpoint['epoch']
            self.log.info(f'Continue training with model from {model_path} at epoch {epoch}')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.to(self.device)
        else:
            epoch = 0

        self.train(epoch)

    def train_with_weights(self, prev_stage_dir):
        assert io.folder_exist(prev_stage_dir), f'cannot find folder {prev_stage_dir}'
        model_path = self.get_latest_model_path(self.cfg.test.with_best, prev_stage_dir)
        assert io.is_non_zero_file(model_path), f'cannot find file {model_path}'

        checkpoint = torch.load(model_path, map_location=self.device)
        epoch = checkpoint['epoch']
        self.log.info(f'training with previous stage model from {model_path} at epoch {epoch}')
        model_dict = self.model.state_dict()
        ckp_dict = checkpoint['model_state_dict']
        pretrained_dict = {k: v for k, v in ckp_dict.items() if k.startswith(tuple(self.cfg.model.pretrained_module))}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)

        self.train()
