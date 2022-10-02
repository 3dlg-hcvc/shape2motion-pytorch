import numpy as np
from enum import Enum
from numpy import linalg as LA
import matplotlib.pyplot as plt
import trimesh
import os

from tools.visualizations import Renderer
from tools.utils import io

# TODO move to one location
class JointType(Enum):
    ROT = 1
    TRANS = 2
    BOTH = 3

Green = np.array([
    [44, 160, 44, 255],
    [152, 223, 138, 255],
]) / 255.0

Yellow = np.array([
    [255, 127, 14, 255],
    [255, 187, 120, 255],
]) / 255.0

Red = np.array([
    [214, 39, 40, 255],
    [255, 152, 150, 255],
]) / 255.0

class Visualizer(Renderer):
    def __init__(self, vertices=None, faces=None, colors=None, normals=None, mask=None):
        super().__init__(vertices, faces, colors, normals, mask)

    def view_stage1_input(self, instance_data):
        input_pts = instance_data['input_pts'][:]
        input_xyz = input_pts[:, :3]
        input_normals = input_pts[:, 6:9]
        anchor_pts_idx = instance_data['anchor_pts'][:]
        joint_direction_cat = instance_data['joint_direction_cat'][:]
        joint_direction_reg = instance_data['joint_direction_reg'][:]
        joint_origin_reg = instance_data['joint_origin_reg'][:]
        joint_type = instance_data['joint_type'][:]
        joint_all_directions = instance_data['joint_all_directions'][:]
        gt_joints = instance_data['gt_joints'][:]
        gt_proposals = instance_data['gt_proposals'][:].astype(np.float32)
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
        viewer.add_trimesh_arrows(joint_origins, joint_directions, colors=joint_colors, radius=0.005, length=0.01)
        viewer.show()

        viewer.reset()

        viewer.add_geometry(vertices=input_xyz, normals=input_normals, mask=mask.astype(int))
        joint_direction_cat = joint_direction_cat[anchor_pts_idx.astype(bool)] - 1
        joint_direction_reg = joint_direction_reg[anchor_pts_idx.astype(bool), :]
        joint_directions = joint_all_directions[joint_direction_cat.astype(int), :] + joint_direction_reg

        viewer.add_trimesh_arrows(anchor_pts_xyz, joint_directions, radius=0.005, length=0.01)
        viewer.show()

    def view_stage1_output(self, pred_cfg, proposal__downsample=1, joint_downsample=2):
        for i in range(0, pred_cfg.part_proposals.shape[0], proposal__downsample):
            pred_viewer = Renderer(vertices=self.vertices, mask=pred_cfg.part_proposals[i].astype(int))
            anchor_mask = pred_cfg.anchor_mask > 0
            pred_motions = pred_cfg.motions[anchor_mask, :]
            pred_joint_origins = pred_motions[:, :3]
            pred_joint_directions = pred_motions[:, 3:6]
            pred_joint_directions = pred_joint_directions / LA.norm(pred_joint_directions, axis=1).reshape(-1,1)
            pred_joint_type = pred_motions[:, 6]

            pred_joint_colors = np.zeros((pred_joint_type.shape[0], 4))
            pred_joint_colors[pred_joint_type == JointType.ROT.value] = [1.0, 0.0, 0.0, 1.0]
            pred_joint_colors[pred_joint_type == JointType.TRANS.value] = [0.0, 0.0, 1.0, 1.0]
            pred_joint_colors[pred_joint_type == JointType.BOTH.value] = [0.0, 1.0, 0.0, 1.0]

            pred_viewer.add_trimesh_arrows(pred_joint_origins[::joint_downsample], pred_joint_directions[::joint_downsample], colors=pred_joint_colors[::joint_downsample], radius=0.005, length=0.2)
            pred_viewer.show(window_name=f'pred_{i}')

    def view_stage2_input(self, gt_cfg, pred_cfg, proposal__downsample=1, joint_downsample=2):
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

    def view_stage2_output(self, gt_cfg, pred_cfg, joint_downsample=2):
        # gt
        # part_proposal
        # motions
        # scores
        # anchor_mask

        # pred
        # part_proposal
        # motions
        # scores
        # anchor_mask

        gt_viewer = Renderer(vertices=self.vertices, mask=gt_cfg.part_proposal.astype(int))
        anchor_mask = gt_cfg.anchor_mask > 0
        gt_motions = gt_cfg.motions[anchor_mask, :]
        gt_joint_origins = gt_motions[:, :3]
        gt_joint_directions = gt_motions[:, 3:6]
        gt_joint_directions = gt_joint_directions / LA.norm(gt_joint_directions, axis=1).reshape(-1,1)
        gt_joint_type = gt_motions[:, 6]

        cm = plt.get_cmap('jet')
        gt_joint_scores = gt_cfg.scores[anchor_mask]
        gt_joint_colors = cm(gt_joint_scores)

        gt_viewer.add_trimesh_arrows(gt_joint_origins[::joint_downsample], gt_joint_directions[::joint_downsample], colors=gt_joint_colors[::joint_downsample], radius=0.005, length=0.2)
        gt_viewer.show(window_name=f'gt')


        pred_viewer = Renderer(vertices=self.vertices, mask=pred_cfg.part_proposal.astype(int))
        anchor_mask = pred_cfg.anchor_mask > 0
        pred_motions = pred_cfg.motions[anchor_mask, :]
        pred_joint_origins = pred_motions[:, :3]
        pred_joint_directions = pred_motions[:, 3:6]
        pred_joint_directions = pred_joint_directions / LA.norm(pred_joint_directions, axis=1).reshape(-1,1)
        pred_joint_type = pred_motions[:, 6]

        cm = plt.get_cmap('jet')
        pred_joint_scores = pred_cfg.scores[anchor_mask]
        pred_joint_colors = cm(pred_joint_scores)

        pred_viewer.add_trimesh_arrows(pred_joint_origins[::joint_downsample], pred_joint_directions[::joint_downsample], colors=pred_joint_colors[::joint_downsample], radius=0.005, length=0.2)
        pred_viewer.show(window_name=f'pred')

    def view_stage3_input(self, cfg):
        input_pts = cfg['input_pts']
        part_proposal = cfg['part_proposal']
        good_motion = cfg['motions']
        anchor_mask = cfg['anchor_mask']

        for i in range(3):
            gt_viewer = Renderer(vertices=input_pts[i, :, :3], mask=part_proposal.astype(int))
            gt_joint_origin = good_motion[:3]
            gt_joint_direction = good_motion[3:6]
            gt_joint_direction = gt_joint_direction / LA.norm(gt_joint_direction)
            gt_joint_type = good_motion[6]

            gt_viewer.add_trimesh_arrows([gt_joint_origin], [gt_joint_direction], radius=0.005, length=0.2)
            gt_viewer.show(window_name=f'mv_pcd_{i}')

    def view_stage3_output(self, gt_cfg, pred_cfg):
        # gt
        # part_proposal

        # pred
        # part_proposal
        gt_viewer = Renderer(vertices=self.vertices, mask=gt_cfg.part_proposal.astype(int))
        gt_viewer.show(window_name=f'gt')

        pred_viewer = Renderer(vertices=self.vertices, mask=pred_cfg.part_proposal.astype(int))
        pred_viewer.show(window_name=f'pred')

    def view_nms_output(self, joints):
        pred_viewer = Renderer(vertices=self.vertices, mask=self.mask)
        joint_directions = joints[:, 3:6]

        mask = np.any(joint_directions, axis=1)
        joints = joints[mask, :]
        joint_origins = joints[:, :3]
        joint_directions = joints[:, 3:6]
        joint_directions = joint_directions / np.linalg.norm(joint_directions, axis=1).reshape(-1, 1)
        
        joint_types = joints[:, 6]
        joint_colors = np.zeros((len(joint_types), 4))
        for i, joint_type in enumerate(joint_types):
            if joint_type == JointType.ROT.value:
                joint_colors[i] = [1.0, 0.0, 0.0, 1.0]
            elif joint_type == JointType.TRANS.value:
                joint_colors[i] = [0.0, 0.0, 1.0, 1.0]
            elif joint_type == JointType.BOTH.value:
                joint_colors[i] = [0.0, 1.0, 0.0, 1.0]

        pred_viewer.add_trimesh_arrows(joint_origins, joint_directions, colors=joint_colors, length=0.4)
        pred_viewer.show(window_name=f'pred', non_block=False)

    def view_evaluation_result_each(self, gt_cfg, pred_cfg):
        # part_proposal
        # joints

        part_proposal = gt_cfg.part_proposal
        joint = gt_cfg.joint
        gt_viewer = Renderer(vertices=self.vertices, mask=part_proposal.astype(int))
        joint_origins = joint[:3]
        joint_directions = joint[3:6]
        joint_directions = joint_directions / np.linalg.norm(joint_directions)
        
        joint_type = joint[6]
        joint_colors = np.zeros((1, 4))
        if joint_type == JointType.ROT.value:
            joint_colors[0] = [1.0, 0.0, 0.0, 1.0]
        elif joint_type == JointType.TRANS.value:
            joint_colors[0] = [0.0, 0.0, 1.0, 1.0]
        elif joint_type == JointType.BOTH.value:
            joint_colors[0] = [0.0, 1.0, 0.0, 1.0]

        gt_viewer.add_trimesh_arrows([joint_origins], [joint_directions], colors=joint_colors, length=0.4)
        gt_viewer.show(window_name='gt', non_block=True)


        part_proposal = pred_cfg.part_proposal
        joint = pred_cfg.joint
        gt_viewer = Renderer(vertices=self.vertices, mask=part_proposal.astype(int))
        joint_origins = joint[:3]
        joint_directions = joint[3:6]
        joint_directions = joint_directions / np.linalg.norm(joint_directions)
        
        joint_type = joint[6]
        joint_colors = np.zeros((1, 4))
        if joint_type == JointType.ROT.value:
            joint_colors[0] = [1.0, 0.0, 0.0, 1.0]
        elif joint_type == JointType.TRANS.value:
            joint_colors[0] = [0.0, 0.0, 1.0, 1.0]
        elif joint_type == JointType.BOTH.value:
            joint_colors[0] = [0.0, 1.0, 0.0, 1.0]

        gt_viewer.add_trimesh_arrows([joint_origins], [joint_directions], colors=joint_colors, length=0.4)
        gt_viewer.show(window_name='pred', non_block=True)

    def compute_angle(self, pred_joint_dir, gt_joint_dir):
        a1 = np.arccos(
            np.clip(np.dot(pred_joint_dir, gt_joint_dir) / (LA.norm(pred_joint_dir) * LA.norm(gt_joint_dir)), -1, 1))
        a2 = np.arccos(
            np.clip(np.dot(-pred_joint_dir, gt_joint_dir) / (LA.norm(-pred_joint_dir) * LA.norm(gt_joint_dir)), -1, 1))
        return min(a1, a2)

    def compute_dist(self, pred_origin, gt_joint):
        q1 = gt_joint[:3]
        q2 = gt_joint[:3] + gt_joint[3:6]
        vec1 = q2 - q1
        vec2 = pred_origin - q1
        dist = LA.norm(np.cross(vec1, vec2)) / LA.norm(vec1)
        return dist

    def view_evaluation_result(self, gt_cfg, pred_cfg, output_dir='viz'):
        part_proposals = gt_cfg.part_proposals
        joints = gt_cfg.joints
        object_id = gt_cfg.object_id
        mask = np.zeros(part_proposals.shape[1])
        for i in range(part_proposals.shape[0]):
            mask[part_proposals[i, :]] = (i+1)
        colors = Renderer.colors_from_mask(mask.astype(int), empty_first=True, color_map='Set2')
        gt_viewer = Renderer(vertices=self.vertices, colors=colors)
        joint_origins = joints[:, :3]
        joint_directions = joints[:, 3:6]
        joint_directions = joint_directions / np.linalg.norm(joint_directions, axis=1).reshape(-1, 1)
        
        joint_types = joints[:, 6]
        joint_colors = np.zeros((len(joint_types), 4))
        head_colors = np.zeros((len(joint_types), 4))
        curls_colors = np.zeros((len(joint_types), 4))
        curls_head_colors = np.zeros((len(joint_types), 4))
        origin_colors = np.zeros((len(joint_types), 4))
        for i, joint_type in enumerate(joint_types):
            joint_colors[i] = Green[0]
            head_colors[i] = Green[1]
            curls_colors[i] = Green[0]
            curls_head_colors[i] = Green[1]
            origin_colors[i] = Green[0]

        gt_arrows = gt_viewer.get_trimesh_arrows(joint_origins, joint_directions, origin_colors=origin_colors, colors=joint_colors, head_colors=head_colors, length=0.4, joint_types=joint_types, curls_colors=curls_colors, curls_head_colors=curls_head_colors)
        
        io.ensure_dir_exists(os.path.join(output_dir, 'gt'))
        arrows = trimesh.util.concatenate(gt_arrows)
        arrows.export(os.path.join(output_dir, 'gt', f'{object_id}_arrows.ply'))
        gt_viewer.export(os.path.join(output_dir, 'gt', f'{object_id}.ply'))

        part_proposals = pred_cfg.part_proposals.astype(bool)
        joints = pred_cfg.joints
        gt_joints = pred_cfg.gt_joints
        object_id = pred_cfg.object_id
        mask = np.zeros(part_proposals.shape[1])
        for i in range(part_proposals.shape[0]):
            mask[part_proposals[i, :]] = (i+1)
        colors = Renderer.colors_from_mask(mask.astype(int), empty_first=True, color_map='Set2')
        pred_viewer = Renderer(vertices=self.vertices, colors=colors)
        joint_origins = joints[:, :3]
        joint_directions = joints[:, 3:6]
        joint_directions = joint_directions / np.linalg.norm(joint_directions, axis=1).reshape(-1, 1)

        gt_joint_origins = gt_joints[:, :3]
        gt_joint_directions = gt_joints[:, 3:6]
        gt_joint_directions = gt_joint_directions / np.linalg.norm(gt_joint_directions, axis=1).reshape(-1, 1)
        
        joint_types = gt_joints[:, 6]
        joint_colors = np.zeros((len(joint_types), 4))
        head_colors = np.zeros((len(joint_types), 4))
        curls_colors = np.zeros((len(joint_types), 4))
        curls_head_colors = np.zeros((len(joint_types), 4))
        origin_colors = np.zeros((len(joint_types), 4))
        for i, joint_type in enumerate(joint_types):
            if self.compute_angle(joint_directions[i], gt_joint_directions[i]) < 5.0 / 180 * np.pi:
                joint_colors[i] = Green[0]
                head_colors[i] = Green[1]
                curls_colors[i] = Green[0]
                curls_head_colors[i] = Green[1]
            elif self.compute_angle(joint_directions[i], gt_joint_directions[i]) < 10.0 / 180 * np.pi:
                joint_colors[i] = Yellow[0]
                head_colors[i] = Yellow[1]
                curls_colors[i] = Yellow[0]
                curls_head_colors[i] = Yellow[1]
            else:
                joint_colors[i] = Red[0]
                head_colors[i] = Red[1]
                curls_colors[i] = Red[0]
                curls_head_colors[i] = Red[1]

            md = self.compute_dist(joint_origins[i], np.concatenate((gt_joint_origins[i], gt_joint_directions[i])))
            scale = np.linalg.norm(np.amax(self.vertices, axis=0) - np.amin(self.vertices, axis=0))
            if md < scale * 0.125:
                origin_colors[i] = Green[0]
            elif md < scale * 0.25:
                origin_colors[i] = Yellow[0]
            else:
                origin_colors[i] = Red[0]

        pred_arrows = pred_viewer.get_trimesh_arrows(joint_origins, joint_directions, origin_colors=origin_colors, colors=joint_colors, head_colors=head_colors, length=0.4, joint_types=joint_types, curls_colors=curls_colors, curls_head_colors=curls_head_colors)

        arrows = trimesh.util.concatenate(pred_arrows)
        io.ensure_dir_exists(os.path.join(output_dir, 'pred'))
        arrows.export(os.path.join(output_dir, 'pred', f'{object_id}_arrows.ply'))
        pred_viewer.export(os.path.join(output_dir, 'pred', f'{object_id}.ply'))
        pred_viewer.render(os.path.join(output_dir, 'pred', f'{object_id}.gif'), as_gif=True)

    def view_input_color(self, pts, object_id, output_dir):
        m = trimesh.Trimesh(vertices=pts[:, :3], vertex_colors=(pts[:, 3:6]+1) / 2)
        io.ensure_dir_exists(os.path.join(output_dir, 'input'))
        m.export(os.path.join(output_dir, 'input', f'{object_id}.ply'))