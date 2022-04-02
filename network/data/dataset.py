import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tools.utils.constant import Stage

class Shape2MotionDataset(Dataset):
    def __init__(self, data_path, num_points, stage):
        self.h5_data = h5py.File(data_path, 'r')
        self.num_points = num_points
        self.stage = stage

        self.instances = self._load()

    def _load(self):
        instances = []
        for instance_name in self.h5_data.keys():
            if self.stage == Stage.stage1:
                filter_input_list = ['joint_all_directions', 'gt_joints', 'gt_proposals']
                instance_data_h5 = self.h5_data[instance_name]
                instance_data = {}
                for k, v in instance_data_h5.items():
                    if k in filter_input_list:
                        continue
                    instance_data[k] = v[:]
                instances.append({'id': instance_name, 'data': instance_data})
            elif self.stage == Stage.stage2:
                object_data = self.h5_data[instance_name]
                pred_part_proposals = object_data['pred_part_proposals'][:]
                for proposal_idx in range(len(pred_part_proposals)):
                    instance_data = {
                        'input_pts': object_data['input_pts'][:],
                        'part_proposal': pred_part_proposals[proposal_idx],
                        'anchor_mask': object_data['pred_anchor_mask'][:],
                        'motion_scores': object_data['motion_scores'][:][proposal_idx]
                    }
                    
                    instances.append({'id': instance_name + f'_{proposal_idx}', 'data': instance_data})

        return instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        instance = self.instances[index]
        id = instance['id']
        instance_data = instance['data']

        input_pts = torch.tensor(instance_data['input_pts'], dtype=torch.float32)
        gt_dict = {}
        for k, v in instance_data.items():
            if k == 'input_pts':
                continue
            else:
                gt_dict[k] = torch.tensor(v, dtype=torch.float32)

        return input_pts, gt_dict, id
