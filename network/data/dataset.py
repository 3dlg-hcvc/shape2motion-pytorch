import h5py
import logging
import torch
from torch.utils.data import Dataset
from tools.utils.constant import Stage

log = logging.getLogger('Dataset')

class Shape2MotionDataset(Dataset):
    def __init__(self, data_path, num_points, stage):
        self.h5_data = h5py.File(data_path, 'r')
        self.num_points = num_points
        self.stage = stage

        self.index2instance = self._pre_load()

    def _pre_load(self):
        index2instance = {}
        index = 0
        for instance_name in self.h5_data.keys():
            if self.stage == Stage.stage1:
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

    def __getitem__(self, index):
        instance_name = self.index2instance[index]

        if self.stage == Stage.stage1:
            instance_data = self.h5_data[instance_name]
            input_pts = torch.tensor(instance_data['input_pts'][:], dtype=torch.float32)

            gt_dict = {}
            filter_input_list = ['input_pts', 'joint_all_directions', 'gt_joints', 'gt_proposals']
            for k, v in instance_data.items():
                if k in filter_input_list:
                        continue
                else:
                    gt_dict[k] = torch.tensor(v[:], dtype=torch.float32)
        elif self.stage == Stage.stage2:
            components = instance_name.split('_')
            object_instance_name = '_'.join(components[:-1])
            proposal_idx = int(components[-1])

            object_data = self.h5_data[object_instance_name]
            input_pts = torch.tensor(object_data['input_pts'][:], dtype=torch.float32)
            gt_dict = {
                'part_proposal': torch.tensor(object_data['pred_part_proposals'][:][proposal_idx], dtype=torch.float32),
                'anchor_mask': torch.tensor(object_data['pred_anchor_mask'][:], dtype=torch.float32),
                'motion_scores': torch.tensor(object_data['motion_scores'][:][proposal_idx], dtype=torch.float32)
            }

        return input_pts, gt_dict, instance_name
