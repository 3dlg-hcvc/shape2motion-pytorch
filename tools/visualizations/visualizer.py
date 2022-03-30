import numpy as np
from enum import Enum
from numpy import linalg as LA
import matplotlib.pyplot as plt

from tools.visualizations import Renderer

import pdb

# TODO move to one location
class JointType(Enum):
    ROT = 1
    TRANS = 2
    BOTH = 3

class Visualizer(Renderer):
    def __init__(self, vertices=None, faces=None, colors=None, normals=None, mask=None):
        super().__init__(vertices, faces, colors, normals, mask)
        pass

    def view_stage2_input(self, gt_cfg, pred_cfg, proposal__downsample=2, joint_downsample=1):
        # gt_cfg.part_proposals
        # gt_cfg.motions
        
        # pred_cfg.part_proposals
        # pred_cfg.motions
        # pred_cfg.scores
        # pred_cfg.anchor_mask

        for i in range(0, gt_cfg.part_proposals.shape[0], proposal__downsample):
            gt_viewer = Renderer(vertices=self.vertices, mask=gt_cfg.part_proposals[i].astype(int))
            gt_motion = gt_cfg.motions[i]
            gt_joint_origin = gt_motion[:3]
            gt_joint_direction = gt_motion[3:6]
            gt_joint_direction = gt_joint_direction / LA.norm(gt_joint_direction)
            gt_joint_type = gt_motion[6]
            if gt_joint_type == JointType.ROT.value:
                gt_joint_color = [1.0, 0.0, 0.0, 1.0]
            elif gt_joint_type == JointType.TRANS.value:
                gt_joint_color = [0.0, 0.0, 1.0, 1.0]
            elif gt_joint_type == JointType.BOTH.value:
                gt_joint_color = [0.0, 1.0, 0.0, 1.0]

            gt_viewer.add_trimesh_arrows([gt_joint_origin], [gt_joint_direction], colors=[gt_joint_color], length=0.5)
            gt_viewer.show(window_name=f'gt_{i}')

            pred_viewer = Renderer(vertices=self.vertices, mask=pred_cfg.part_proposals[i].astype(int))
            anchor_mask = pred_cfg.anchor_mask > 0
            pred_motions = pred_cfg.motions[anchor_mask, :]
            pred_joint_origins = pred_motions[:, :3]
            pred_joint_directions = pred_motions[:, 3:6]
            pred_joint_directions = pred_joint_directions / LA.norm(pred_joint_directions, axis=1).reshape(-1,1)
            pred_joint_type = pred_motions[:, 6]

            rot_cm = plt.get_cmap('Reds')
            trans_cm = plt.get_cmap('Blues')
            both_cm = plt.get_cmap('Greens')

            pred_joint_scores = pred_cfg.scores[i][anchor_mask]
            pred_rot_joint_colors = rot_cm(pred_joint_scores)
            pred_trans_joint_colors = trans_cm(pred_joint_scores)
            pred_both_joint_colors = both_cm(pred_joint_scores)
            pred_joint_colors = np.zeros((pred_joint_scores.shape[0], 4))
            rot_mask = pred_joint_type == JointType.ROT.value
            trans_mask = pred_joint_type == JointType.TRANS.value
            both_mask = pred_joint_type == JointType.BOTH.value
            pred_joint_colors[rot_mask] = pred_rot_joint_colors[rot_mask] + np.asarray([0.2, 0.0, 0.0, 0.0])
            pred_joint_colors[trans_mask] = pred_trans_joint_colors[trans_mask] + np.asarray([0.0, 0.0, 0.2, 0.0])
            pred_joint_colors[both_mask] = pred_both_joint_colors[both_mask] + np.asarray([0.0, 0.2, 0.0, 0.0])
            pred_joint_colors = np.clip(pred_joint_colors, 0.0, 1.0)

            pred_viewer.add_trimesh_arrows(pred_joint_origins[::joint_downsample], pred_joint_directions[::joint_downsample], colors=pred_joint_colors[::joint_downsample], radius=0.005, length=0.2)
            pred_viewer.show(window_name=f'pred_{i}')


            
