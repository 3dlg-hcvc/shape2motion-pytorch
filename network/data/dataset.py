import h5py
import logging
import torch
from torch.utils.data import Dataset
from tools.utils.constant import Stage
import numpy as np

log = logging.getLogger('Dataset')


class Shape2MotionDataset(Dataset):
    def __init__(self, data_path, num_points, stage, augmentation_cfg=None):
        self.h5_data = h5py.File(data_path, 'r')
        self.num_points = num_points
        self.stage = stage
        self.augmentation_cfg = augmentation_cfg

        self.index2instance = self._pre_load()

    def _pre_load(self):
        index2instance = {}
        index = 0
        for instance_name in self.h5_data.keys():
            if self.stage == Stage.stage1 or self.stage == Stage.stage3:
                index2instance[index] = instance_name
                index += 1
            elif self.stage == Stage.stage2:
                object_data = self.h5_data[instance_name]
                pred_part_proposals = object_data['pred_part_proposals'][:]
                for proposal_idx in range(len(pred_part_proposals)):
                    index2instance[index] = instance_name + f'_{proposal_idx}'
                    index += 1
        log.info(f'Dataset contains {len(index2instance)} instance data')
        return index2instance

    def __len__(self):
        return len(self.index2instance)

    def data_augment(self, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0],
                              [0, 0, 1]])  # rotation
        return m

    def __getitem__(self, index):
        instance_name = self.index2instance[index]

        if self.augmentation_cfg is not None:
            m = self.data_augment(
                self.augmentation_cfg.jitter,
                self.augmentation_cfg.flip,
                self.augmentation_cfg.rotate,
            )
            m = torch.from_numpy(m.astype(np.float32))

        if self.stage == Stage.stage1 or self.stage == Stage.stage3:
            instance_data = self.h5_data[instance_name]

            input_pts = torch.from_numpy(instance_data['input_pts'][:])

            gt_dict = {}
            filter_input_list = ['input_pts', 'joint_all_directions', 'gt_joints', 'gt_proposals', 'pred_motion_scores',
                                 'good_motion']
            for k, v in instance_data.items():
                if k in filter_input_list:
                    continue
                else:
                    gt_dict[k] = torch.from_numpy(v[:])

            if self.augmentation_cfg is not None:
                input_pts[:, :3] = torch.matmul(input_pts[:, :3], m)
                inv_trans_m = torch.inverse(m).transpose(0, 1)
                input_pts[:, 6:9] = torch.matmul(input_pts[:, 6:9], inv_trans_m)
                if self.augmentation_cfg.color:
                    color_rand = torch.randn(3) * 0.05
                    input_pts[:, 3:6] += color_rand

                if self.stage == Stage.stage1:
                    gt_dict['joint_origin_reg'] = torch.matmul(gt_dict['joint_origin_reg'], m)
                else:
                    gt_dict['motion_regression'][:3] = torch.matmul(gt_dict['motion_regression'][:3], m)
                    gt_dict['moved_pcds'][:, :, :3] = torch.matmul(gt_dict['moved_pcds'][:, :, :3], m)

                    gt_dict['moved_pcds'][:, :, 6:9] = torch.matmul(gt_dict['moved_pcds'][:, :, 6:9], inv_trans_m)

                    if self.augmentation_cfg.color:
                        gt_dict['moved_pcds'][:, :, 3:6] += color_rand

        elif self.stage == Stage.stage2:
            components = instance_name.split('_')
            object_instance_name = '_'.join(components[:-1])
            proposal_idx = int(components[-1])

            object_data = self.h5_data[object_instance_name]
            input_pts = torch.from_numpy(object_data['input_pts'][:])
            gt_dict = {
                'part_proposal': torch.from_numpy(object_data['pred_part_proposals'][:][proposal_idx]),
                'anchor_mask': torch.from_numpy(object_data['pred_anchor_mask'][:]),
                'motion_scores': torch.from_numpy(object_data['motion_scores'][:][proposal_idx])
            }

            if self.augmentation_cfg is not None:
                input_pts[:, :3] = torch.matmul(input_pts[:, :3], m)
                inv_trans_m = torch.inverse(m).transpose(0, 1)
                input_pts[:, 6:9] = torch.matmul(input_pts[:, 6:9], inv_trans_m)
                if self.augmentation_cfg.color:
                    color_rand = torch.randn(3) * 0.05
                    input_pts[:, 3:6] += color_rand

        return input_pts, gt_dict, instance_name
