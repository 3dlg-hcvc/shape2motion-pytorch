from cmath import pi
import enum
import h5py
import numpy as np
from types import SimpleNamespace
from numpy import linalg as LA

from tools.visualizations import Visualizer
from tools.utils import io
from tools.utils.constant import JointType
import pdb

class NMS:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stage1 = cfg.stage1
        self.stage2 = cfg.stage2
        self.stage3 = cfg.stage3

    def process(self, output, data_set):
        if data_set == 'train':
            assert io.is_non_zero_file(self.stage1.train), OSError(f'Cannot find file {self.stage1.train}')
            stage1_output = h5py.File(self.stage1.train, 'r')

            assert io.is_non_zero_file(self.stage2.train), OSError(f'Cannot find file {self.stage2.train}')
            stage2_output = h5py.File(self.stage2.train, 'r')

            assert io.is_non_zero_file(self.stage3.train), OSError(f'Cannot find file {self.stage3.train}')
            stage3_output = h5py.File(self.stage3.train, 'r')

            # get predictions for one object
            for object_name in stage1_output.keys():
                instance_names = []
                for instance_name in stage3_output.keys():
                    components = instance_name.split('_')
                    object_instance_name = '_'.join(components[:-1])
                    if object_instance_name == object_name:
                        instance_names.append(instance_name)

            for object_name in stage1_output.keys():
                stage1_instance = stage1_output[object_name]
                input_pts = stage1_instance['input_pts'][:]
                input_xyz = input_pts[:, :3]
                pred_anchor_mask = stage1_instance['pred_anchor_mask'][:].astype(bool)
                pred_motions = stage1_instance['pred_motions'][:]
                boxes = np.zeros((len(instance_names), 7))
                # pick the part proposals
                for i, instance_name in enumerate(instance_names):
                    stage2_instance = stage2_output[instance_name]
                    pred_motion_scores = stage2_instance['pred_motion_scores'][:]
                    stage3_instance = stage3_output[instance_name]
                    pred_part_proposal = stage3_instance['part_proposal'][:].astype(bool)

                    part_pts = input_xyz[pred_part_proposal, :]
                    min_bound = np.amin(part_pts, axis=0)
                    max_bound = np.amax(part_pts, axis=0)
                    score = np.amax(pred_motion_scores)
                    boxes[i, :3] = min_bound
                    boxes[i, 3:6] = max_bound
                    boxes[i, 6] = score
                part_pick = self.nms(boxes)
                assert(len(part_pick) > 0), 'No part proposal picked in nms'

                selected_joints = []
                for idx in part_pick:
                    instance_name = instance_names[idx]
                    
                    stage2_instance = stage2_output[instance_name]
                    pred_motion_scores = stage2_instance['pred_motion_scores'][:]

                    tmp_pred_motion_scores = pred_motion_scores[pred_anchor_mask]
                    tmp_pred_motion_mask = np.where(tmp_pred_motion_scores > self.cfg.score_threshold)[0]
                    tmp_pred_motion_scores = tmp_pred_motion_scores[tmp_pred_motion_mask]
                    tmp_pred_motions = pred_motions[pred_anchor_mask, :][tmp_pred_motion_mask, :]
                    tmp_pred_motions_directions = tmp_pred_motions[:, 3:6]
                    tmp_pred_motions_directions = tmp_pred_motions_directions / LA.norm(tmp_pred_motions_directions, axis=1).reshape(-1, 1)
                    tmp_pred_motions_types = tmp_pred_motions[:, 6]
                    joint_pick = self.nms_joints(tmp_pred_motions_directions, tmp_pred_motions_types, tmp_pred_motion_scores)
                    selected_joints.append(tmp_pred_motion_mask[joint_pick])

                pred_part_proposal_all = np.zeros(input_xyz.shape[0])
                pred_joints_all = None
                for i, pick_idx in enumerate(part_pick):
                    instance_name = instance_names[pick_idx]

                    stage3_instance = stage3_output[instance_name]
                    pred_part_proposal = stage3_instance['part_proposal'][:].astype(bool)
                    pred_part_proposal_all[pred_part_proposal] = i+1

                    pred_joints_idx = selected_joints[i]
                    pred_joints = pred_motions[pred_anchor_mask, :][pred_joints_idx, :]

                    if pred_joints_all is None:
                        pred_joints_all = pred_joints
                    else:
                        pred_joints_all = np.concatenate((pred_joints_all, pred_joints), axis=0)
                
                if self.cfg.debug:
                    viz = Visualizer(input_xyz, mask=pred_part_proposal_all.astype(int))
                    viz.view_nms_output(pred_joints_all)
            
    def nms(self, boxes):
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        z1 = boxes[:,2]
        x2 = boxes[:,3]
        y2 = boxes[:,4]
        z2 = boxes[:,5]
        score = boxes[:,6]
        area = (x2-x1)*(y2-y1)*(z2-z1)

        I = np.argsort(score)
        pick = []
        while (I.size!=0):
            last = I.size
            i = I[-1]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[I[:last-1]])
            yy1 = np.maximum(y1[i], y1[I[:last-1]])
            zz1 = np.maximum(z1[i], z1[I[:last-1]])
            xx2 = np.minimum(x2[i], x2[I[:last-1]])
            yy2 = np.minimum(y2[i], y2[I[:last-1]])
            zz2 = np.minimum(z2[i], z2[I[:last-1]])

            l = np.maximum(0, xx2-xx1)
            w = np.maximum(0, yy2-yy1)
            h = np.maximum(0, zz2-zz1)

            inter = l*w*h
            o = inter / (area[i] + area[I[:last-1]] - inter)

            I = np.delete(I, np.concatenate(([last-1], np.where(o > self.cfg.overlap_threshold)[0])))

        return pick

    def nms_joints(self, joint_directions, joint_types, scores):
        rot_mask = np.where(joint_types == JointType.ROT.value)[0]
        trans_mask = np.where(joint_types == JointType.TRANS.value)[0]
        rot_joints = joint_directions[rot_mask, :]
        rot_scores = scores[rot_mask]
        trans_joints = joint_directions[trans_mask, :]
        trans_scores = scores[trans_mask]

        I = np.argsort(rot_scores)
        pick = []
        while (I.size!=0):
            last = I.size
            i = I[-1]
            pick.append(i)

            dot_prod = np.dot(rot_joints[I[:last-1]], rot_joints[i])
            angle_threshold = float(self.cfg.angle_threshold) / 180.0 * np.pi
            I = np.delete(I, np.concatenate(([last-1], np.where(dot_prod > angle_threshold)[0])))
        if len(pick) > 0:
            pick = rot_mask[pick]

        if len(trans_joints) > 0:
            pick.append(trans_mask[np.argmax(trans_scores)])

        return pick


