import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tools.utils.constant import Stage

class Shape2MotionDataset(Dataset):
    def __init__(self, data_path, num_points, stage):
        self.f_data = h5py.File(data_path, 'r')
        self.instances = sorted(self.f_data)
        self.num_points = num_points
        self.stage = stage

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        id = self.instances[index]
        ins = self.f_data[id]

        if self.stage == Stage.stage1:
            filter_input_list = ['input_pts', 'joint_all_directions', 'gt_joints', 'gt_proposals']
            
            input_pts = torch.tensor(ins['input_pts'][:], dtype=torch.float32)
            gt_dict = {}
            for k, v in ins.items():
                if k in filter_input_list:
                    continue
                else:
                    gt_dict[k] = torch.tensor(v[:], dtype=torch.float32)

            return input_pts, gt_dict, id
