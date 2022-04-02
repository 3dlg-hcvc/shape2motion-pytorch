import numpy as np
from enum import Enum
from numpy import linalg as LA
import matplotlib.pyplot as plt
import trimesh

from tools.visualizations import Renderer


# TODO move to one location
class JointType(Enum):
    ROT = 1
    TRANS = 2
    BOTH = 3

class Visualizer(Renderer):
    def __init__(self, vertices=None, faces=None, colors=None, normals=None, mask=None):
        super().__init__(vertices, faces, colors, normals, mask)
        pass

    def view_stage1_input(self, instance_data):
        input_pts = instance_data['input_pts'][:]
        input_xyz = input_pts[:, :3]
        input_normals = input_pts[:, 3:]
        anchor_pts_idx = instance_data['anchor_pts'][:]
        joint_direction_cat = instance_data['joint_direction_cat'][:]
        joint_direction_reg = instance_data['joint_direction_reg'][:]
        joint_origin_reg = instance_data['joint_origin_reg'][:]
        joint_type = instance_data['joint_type'][:]
        joint_all_directions = instance_data['joint_all_directions'][:]
        gt_joints = instance_data['gt_joints'][:]
        gt_proposals = instance_data['gt_proposals'][:]
        simmat = instance_data['simmat'][:]

        mask = np.zeros_like(gt_proposals)
        for i in range(len(gt_proposals)):
            mask[i] = gt_proposals[i]*i
        mask = np.sum(mask, axis=0)

        viewer = Renderer(vertices=input_xyz, normals=input_normals, mask=mask.astype(int))
        anchor_pts_xyz = input_xyz[anchor_pts_idx.astype(bool)] + joint_origin_reg[anchor_pts_idx.astype(bool)]
        for i in range(len(anchor_pts_xyz)):
            t = np.eye(4)
            t[:3, 3] = anchor_pts_xyz[i]
            sphere = trimesh.creation.icosphere(radius=0.005, color=[0.0, 1.0, 0.0])
            sphere.apply_transform(t)
            viewer.add_trimesh(sphere)

        joint_origins = gt_joints[:, :3]
        joint_directions = gt_joints[:, 3:6]
        joint_directions = joint_directions / np.linalg.norm(joint_directions, axis=1).reshape(-1, 1)
        
        joint_types = gt_joints[:, 6]
        joint_colors = np.zeros((len(joint_types), 4))
        for i, joint_type in enumerate(joint_types):
            if joint_type == JointType.ROT.value:
                joint_colors[i] = [1.0, 0.0, 0.0, 1.0]
            elif joint_type == JointType.TRANS.value:
                joint_colors[i] = [0.0, 0.0, 1.0, 1.0]
            elif joint_type == JointType.BOTH.value:
                joint_colors[i] = [0.0, 1.0, 0.0, 1.0]

        viewer.add_trimesh_arrows(joint_origins, joint_directions, colors=joint_colors, length=0.4)
        viewer.show()

        viewer.reset()

        viewer.add_geometry(vertices=input_xyz, normals=input_normals, mask=mask.astype(int))
        joint_direction_cat = joint_direction_cat[anchor_pts_idx.astype(bool)] - 1
        joint_direction_reg = joint_direction_reg[anchor_pts_idx.astype(bool), :]
        joint_directions = joint_all_directions[joint_direction_cat.astype(int), :] + joint_direction_reg

        viewer.add_trimesh_arrows(anchor_pts_xyz, joint_directions, radius=0.005, length=0.2)
        viewer.show()


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

            gt_viewer.add_trimesh_arrows([gt_joint_origin], [gt_joint_direction], colors=[gt_joint_color], length=0.4)
            gt_viewer.show(window_name=f'gt_{i}')

            pred_viewer = Renderer(vertices=self.vertices, mask=pred_cfg.part_proposals[i].astype(int))
            anchor_mask = pred_cfg.anchor_mask > 0
            pred_motions = pred_cfg.motions[anchor_mask, :]
            pred_joint_origins = pred_motions[:, :3]
            pred_joint_directions = pred_motions[:, 3:6]
            pred_joint_directions = pred_joint_directions / LA.norm(pred_joint_directions, axis=1).reshape(-1,1)
            pred_joint_type = pred_motions[:, 6]

            cm = plt.get_cmap('jet')

            pred_joint_scores = pred_cfg.scores[i][anchor_mask]
            pred_joint_colors = cm(pred_joint_scores)

            pred_viewer.add_trimesh_arrows(pred_joint_origins[::joint_downsample], pred_joint_directions[::joint_downsample], colors=pred_joint_colors[::joint_downsample], radius=0.005, length=0.2)
            pred_viewer.show(window_name=f'pred_{i}')

    def view_stage2_output(self, gt_cfg, pred_cfg):
        # gt_cfg.part_proposal
        # gt_cfg.motion_scores
        # gt_cfg.anchor_mask
        
        # pred_cfg.motion_scores
        part_mask = gt_cfg.part_proposal.astype(bool)
        anchor_mask = gt_cfg.anchor_mask.astype(bool)
        gt_pts_colors = np.tile([0.6, 0.6, 0.6, 1.0], (part_mask.shape[0], 1))
        gt_pts_colors[part_mask] = [0.1, 0.1, 0.1, 1.0]
        pred_pts_colors = np.copy(gt_pts_colors)

        cm = plt.get_cmap('jet')
        gt_joint_scores = gt_cfg.motion_scores[anchor_mask]
        gt_joint_colors = cm(gt_joint_scores)
        gt_pts_colors[anchor_mask] = gt_joint_colors

        pred_joint_scores = pred_cfg.motion_scores[anchor_mask]
        pred_joint_colors = cm(pred_joint_scores)
        pred_pts_colors[anchor_mask] = pred_joint_colors

        gt_viewer = Renderer(vertices=self.vertices, colors=gt_pts_colors)
        gt_viewer.show(window_name='gt')

        pred_viewer = Renderer(vertices=self.vertices, colors=pred_pts_colors)
        pred_viewer.show(window_name='pred')

            
