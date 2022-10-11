import os
import logging
import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R
import h5py
from enum import Enum
from multiprocessing import Pool, cpu_count
from types import SimpleNamespace

from network import utils
from tools.utils import io
from tools.utils.constant import JointType
from tools.visualizations import Visualizer

from time import time

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger('evaluate')


def get_latest_nms_output_cfg(nms_cfg):
    output_cfg = OmegaConf.create()
    nms_dir = os.path.dirname(nms_cfg.path)
    folder, _ = utils.get_latest_file_with_datetime(nms_dir, '', subdir='', ext='.h5')
    input_dir = os.path.join(nms_dir, folder)
    output_cfg.train = os.path.join(input_dir, 'train_' + nms_cfg.output.nms_result)
    output_cfg.val = os.path.join(input_dir, 'val_' + nms_cfg.output.nms_result)
    output_cfg.test = os.path.join(input_dir, 'test_' + nms_cfg.output.nms_result)
    output_cfg.nms_dir = nms_dir
    return output_cfg


class Evaluation:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.debug:
            log.setLevel(logging.DEBUG)

    def compute_epe(self, pts, pred_joint, gt_joint):
        gt_mv_pts = self.move_pts_with_joint(pts, gt_joint[:3], gt_joint[3:6], gt_joint[6])
        pred_mv_pts = self.move_pts_with_joint(pts, pred_joint[:3], pred_joint[3:6], pred_joint[6])
        epe1 = np.mean(LA.norm(gt_mv_pts - pred_mv_pts, axis=1))

        gt_mv_pts = self.move_pts_with_joint(pts, gt_joint[:3], gt_joint[3:6], gt_joint[6])
        pred_mv_pts = self.move_pts_with_joint(pts, pred_joint[:3], -pred_joint[3:6], pred_joint[6])
        epe2 = np.mean(LA.norm(gt_mv_pts - pred_mv_pts, axis=1))
        return min(epe1, epe2)

    def compute_angle(self, pred_joint_dir, gt_joint_dir):
        a1 = np.arccos(
            np.clip(np.dot(pred_joint_dir, gt_joint_dir) / (LA.norm(pred_joint_dir) * LA.norm(gt_joint_dir)), -1, 1))
        a2 = np.arccos(
            np.clip(np.dot(-pred_joint_dir, gt_joint_dir) / (LA.norm(-pred_joint_dir) * LA.norm(gt_joint_dir)), -1, 1))
        return min(a1, a2) / np.pi * 180.0

    def move_pts_with_joint(self, pts, joint_origin, joint_direction, joint_type, angle=np.pi, trans=1.0):
        if joint_type == JointType.ROT.value:
            move_pts = self.rot3d(pts, joint_origin, joint_direction, angle)
        elif joint_type == JointType.TRANS.value:
            move_pts = self.trans3d(pts, joint_direction, trans)
        elif joint_type == JointType.BOTH.value:
            move_pts = self.rot3d(pts, joint_origin, joint_direction, angle)
            move_pts = self.trans3d(move_pts, joint_direction, trans)
        else:
            log.warn(f'No implementation for the joint type value {joint_type}')
        return move_pts

    def rot3d(self, pts, joint_origin, joint_direction, angle):
        joint_direction = joint_direction / LA.norm(joint_direction)
        rot_mat = R.from_rotvec(angle * joint_direction).as_matrix()
        rot_pts = np.dot(pts - joint_origin, rot_mat.transpose()) + joint_origin
        return rot_pts

    def trans3d(self, pts, joint_direction, trans):
        joint_direction = joint_direction / LA.norm(joint_direction)
        shift_vec = joint_direction * trans
        trans_pts = pts + shift_vec
        return trans_pts

    def compute_dist(self, pred_origin, gt_joint):
        q1 = gt_joint[:3]
        q2 = gt_joint[:3] + gt_joint[3:6]
        vec1 = q2 - q1
        vec2 = pred_origin - q1
        dist = LA.norm(np.cross(vec1, vec2)) / LA.norm(vec1)
        return dist

    def evaluate(self, gt_h5file, pred_h5file, inst_pred_h5file=None, articulation_h5file=None):
        gt_h5 = h5py.File(gt_h5file, 'r')
        pred_h5 = h5py.File(pred_h5file, 'r')
        if inst_pred_h5file is not None:
            inst_pred_h5 = h5py.File(inst_pred_h5file, 'r')

        if articulation_h5file is not None:
            articulation_h5 = h5py.File(articulation_h5file, 'r')

        best_matches = []
        log.debug(pred_h5.keys())
        log.debug(len(gt_h5.keys()))
        for object_id in gt_h5.keys():
            if articulation_h5file is not None:
                object_articulation = articulation_h5['_'.join(object_id.split('_')[:-1])]
                parts_closed = object_articulation['part_closed'][:]
            best_match = {
                'object_id': object_id,
                'iou': [],
                'epe': [],
                'md': [],
                'oe': [],
                'ta': [],
                'part_matches': {},
                'joint_matches': {},
                'M': [],
                'MA': [],
                'MAO': [],
                'num_gt_parts': 0,
                'num_pred_parts': 0,
                'num_gt_joints': 0,
                'num_pred_joints': 0,
            }

            if inst_pred_h5file is not None:
                raw_name = '_'.join(object_id.split('_')[:-1])
                inst_pred_h5_inst = inst_pred_h5[raw_name]

            gt_object = gt_h5[object_id]
            input_pts = gt_object['input_pts'][:]
            if inst_pred_h5file is not None and inst_pred_h5_inst.attrs['has_input']:
                input_pts = np.concatenate((input_pts, inst_pred_h5_inst['eval_add_pts'][:]), axis=0)

            input_xyz = input_pts[:, :3]
            
            if inst_pred_h5file is not None:
                num_parts = inst_pred_h5_inst.attrs['numParts']
                if inst_pred_h5_inst.attrs['has_input']:
                    gt_proposals = np.zeros((num_parts + 1, input_xyz.shape[0]))
                    gt_inst_mask = gt_object['gt_proposals'][:]
                    mask = np.zeros_like(gt_inst_mask)
                    for i in range(len(gt_inst_mask)):
                        mask[i] = gt_inst_mask[i]*i
                    gt_inst_mask = np.sum(mask, axis=0)

                    part_instance_masks = np.concatenate((gt_inst_mask, inst_pred_h5_inst['eval_add_vertex_inst'][:]))
                    part_instance_masks[part_instance_masks < 0] = 0
                    try:
                        assert num_parts == np.unique(part_instance_masks).shape[0] - 1
                    except:
                        import pdb
                        pdb.set_trace()
                
                    for i in range(num_parts):
                        gt_proposals[i + 1] = part_instance_masks == (i + 1)
                else:
                    gt_proposals = gt_object['gt_proposals'][:]
            else:
                gt_proposals = gt_object['gt_proposals'][:]
                num_parts = gt_proposals.shape[0] - 1

            gt_proposals = gt_proposals[1:, :].astype(bool)
            gt_joints = gt_object['gt_joints'][:]
            turn_idx = np.where(np.sum(gt_proposals, axis=1) == 0)[0]

            best_match['num_gt_parts'] = num_parts
            best_match['num_gt_joints'] = gt_joints.shape[0]

            if object_id not in pred_h5.keys() or (inst_pred_h5file is not None and not inst_pred_h5_inst.attrs['has_input']):
                best_match['num_pred_parts'] = 0
                best_match['num_pred_joints'] = 0
                best_matches.append(best_match)
                continue

            pred_object = pred_h5[object_id]
            pred_part_proposals = pred_object['pred_part_proposal'][:]
            if inst_pred_h5file is not None:
                pred_part_proposals = np.concatenate((pred_part_proposals, np.zeros_like(inst_pred_h5_inst['eval_add_vertex_inst'][:])))
            pred_joints = pred_object['pred_joints'][:]
            pred_scores = pred_object['pred_scores'][:]
            pred_joints_map = pred_object['pred_joints_map'][:]
            best_match['num_pred_parts'] = len(np.unique(pred_part_proposals)) - 1
            best_match['num_pred_joints'] = np.sum(np.any(pred_joints, axis=1))

            best_parts = np.ones(gt_proposals.shape[0]) * -1.0
            best_ious = np.ones(gt_proposals.shape[0]) * -1.0
            for part_idx in range(gt_proposals.shape[0]):
                if articulation_h5file is not None and self.cfg.eval_closed:
                    part_closed_state = parts_closed[part_idx]
                elif articulation_h5file is not None:
                    part_closed_state = ~parts_closed[part_idx]
                if articulation_h5file is not None and part_closed_state:
                    best_match['num_gt_parts'] -= 1
                    best_match['num_gt_joints'] -= 1
                    continue

                is_turn = np.where(turn_idx == part_idx)[0].size > 0
                if is_turn:
                    continue

                gt_proposal = gt_proposals[part_idx, :]
                best_part = -1
                best_iou = -1
                for pred_part_idx in range(np.unique(pred_part_proposals).shape[0] - 1):
                    if pred_part_idx in best_parts:
                        continue
                    pred_part_proposal = pred_part_proposals == (pred_part_idx + 1)
                    inter = np.sum(np.logical_and(pred_part_proposal, gt_proposal))
                    outer = np.sum(np.logical_or(pred_part_proposal, gt_proposal))
                    iou = inter / (outer + 1.0e-9)
                    if iou > self.cfg.iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_part = pred_part_idx
                best_parts[part_idx] = best_part
                best_ious[part_idx] = best_iou
                if best_part < 0:
                    continue

                best_match['iou'].append(best_iou)
                best_match['part_matches'][part_idx] = best_part
                have_turn = np.where(turn_idx == part_idx + 1)[0].size > 0
                pred_joints_idx = np.where(pred_joints_map == best_part)[0]
                pred_part_joints_scores = pred_scores[pred_joints_idx]
                pred_part_joints_sorted_idx = np.argsort(pred_part_joints_scores)[::-1]
                pred_part_joints_sorted_idx = pred_joints_idx[pred_part_joints_sorted_idx]
                pred_part_joints = pred_joints[pred_part_joints_sorted_idx, :]
                gt_joint = gt_joints[part_idx]
                selected_joint = None
                for j, joint in enumerate(pred_part_joints):
                    if articulation_h5file is not None and part_closed_state:
                        best_match['num_pred_parts'] -= 1
                        best_match['num_pred_joints'] -= 1
                        continue

                    if not np.any(joint) or part_idx in best_match['joint_matches'].keys():
                        continue
                    md = None
                    oe = None
                    ta = None
                    epe = None
                    if part_idx not in best_match['joint_matches'].keys():
                        md = self.compute_dist(joint[:3], gt_joint)
                        oe = self.compute_angle(joint[3:6], gt_joint[3:6])
                        ta = joint[6] == gt_joint[6]
                        epe = self.compute_epe(input_xyz[gt_proposal, :], joint, gt_joint)
                        selected_joint = gt_joint
                    if have_turn:
                        if part_idx + 1 not in best_match['joint_matches'].keys():
                            gt_joint2 = gt_joints[part_idx + 1]
                            md2 = self.compute_dist(joint[:3], gt_joint2)
                            oe2 = self.compute_angle(joint[3:6], gt_joint2[3:6])
                            ta2 = joint[6] == gt_joint2[6]
                            epe2 = self.compute_epe(input_xyz[gt_proposal, :], joint, gt_joint2)
                            if oe is not None and epe2 < epe:
                                best_match['joint_matches'][part_idx + 1] = pred_part_joints_sorted_idx[j]
                                md = md2
                                oe = oe2
                                ta = ta2
                                epe = epe2
                                selected_joint = gt_joint2
                            elif oe is not None and epe2 >= epe:
                                best_match['joint_matches'][part_idx] = pred_part_joints_sorted_idx[j]
                            else:
                                md = md2
                                oe = oe2
                                ta = ta2
                                epe = epe2
                                selected_joint = gt_joint2
                                best_match['joint_matches'][part_idx + 1] = pred_part_joints_sorted_idx[j]
                        else:
                            best_match['joint_matches'][part_idx] = pred_part_joints_sorted_idx[j]
                    else:
                        best_match['joint_matches'][part_idx] = pred_part_joints_sorted_idx[j]
                    if md is not None:
                        if ta == 1:
                            if selected_joint[6] == JointType.ROT.value:
                                best_match['md'].append(md)
                            best_match['oe'].append(oe)
                            best_match['epe'].append(epe)
                        best_match['ta'].append(ta)
                        if ta == 1:
                            best_match['M'].append(pred_part_joints_sorted_idx[j])
                            if oe < 10.0:
                                best_match['MA'].append(pred_part_joints_sorted_idx[j])
                                scale = np.linalg.norm(np.amax(input_xyz, axis=0) - np.amin(input_xyz, axis=0))
                                if md < scale * 0.25 or selected_joint[6] == JointType.TRANS.value:
                                    best_match['MAO'].append(pred_part_joints_sorted_idx[j])

            best_matches.append(best_match)
            if self.cfg.debug:
                gt_cfg = {}
                gt_part_indices = np.asarray(list(best_match['part_matches'].keys()))
                gt_joint_indices = np.asarray(list(best_match['joint_matches'].keys()))
                if len(gt_part_indices) == 0 or len(gt_joint_indices) == 0:
                    continue

                gt_cfg['part_proposals'] = gt_proposals
                gt_cfg['joints'] = gt_joints
                gt_cfg['object_id'] = object_id
                gt_cfg = SimpleNamespace(**gt_cfg)

                pred_cfg = {}
                pred_part_indices = np.asarray(list(best_match['part_matches'].values()))
                pred_joint_indices = np.asarray(list(best_match['joint_matches'].values()))
                matched_pred_part_proposals = np.zeros((pred_part_indices.shape[0], input_xyz.shape[0]))
                for i in range(matched_pred_part_proposals.shape[0]):
                    matched_pred_part_proposals[i] = pred_part_proposals == (pred_part_indices[i] + 1)
                matched_pred_joints = pred_joints[pred_joint_indices, :]
                matched_gt_joints = gt_joints[gt_joint_indices, :]
                pred_cfg['part_proposals'] = matched_pred_part_proposals
                pred_cfg['joints'] = matched_pred_joints
                pred_cfg['gt_joints'] = matched_gt_joints
                pred_cfg['object_id'] = object_id
                pred_cfg = SimpleNamespace(**pred_cfg)

                input_xyz = gt_object['input_pts'][:][:, :3]
                viz = Visualizer(input_xyz)
                output_dir = os.path.join(self.cfg.output_dir, 'viz')
                viz.view_evaluation_result(gt_cfg, pred_cfg, output_dir=io.to_abs_path(output_dir, get_original_cwd()))
                viz.view_input_color(gt_object['input_pts'][:], object_id, output_dir=io.to_abs_path(output_dir, get_original_cwd()))

        eval_results = {
            'iou': [],
            'epe': [],
            'md': [],
            'oe': [],
            'ta': [],
            'part_recall': [],
            'joint_recall': [],
            'part_precision': [],
            'joint_precision': [],
            'part_f1': [],
            'joint_f1': [],
            'pred_part_sum': [],
            'gt_part_sum': [],
            'pred_joint_sum': [],
            'gt_joint_sum': [],
            'match_part_sum': [],
            'match_joint_sum': [],

            'M_joint_recall': [],
            'MA_joint_recall': [],
            'MAO_joint_recall': [],
            'M_joint_precision': [],
            'MA_joint_precision': [],
            'MAO_joint_precision': [],
            'M_joint_f1': [],
            'MA_joint_f1': [],
            'MAO_joint_f1': [],
            
            'M_num': [],
            'MA_num': [],
            'MAO_num': [],
        }
        names = []
        for best_match in best_matches:
            if best_match['num_gt_parts'] == 0:
                continue
            # if best_match['object_id'].split('_')[0] == 'motor':
            eval_results['iou'] += best_match['iou']
            eval_results['epe'] += best_match['epe']
            eval_results['md'] += best_match['md']
            eval_results['oe'] += best_match['oe']
            eval_results['ta'] += best_match['ta']

            part_recall = len(best_match['part_matches']) / best_match['num_gt_parts']
            joint_recall = len(best_match['joint_matches']) / best_match['num_gt_joints']

            if best_match['num_pred_parts'] > 0:
                part_precision = len(best_match['part_matches']) / best_match['num_pred_parts']
            else:
                part_precision = 0

            if best_match['num_pred_joints'] > 0:
                joint_precision = len(best_match['joint_matches']) / best_match['num_pred_joints']
            else:
                joint_precision = 0
            
            if (part_precision + part_recall) > 0:
                part_f1 = 2 * (part_precision * part_recall) / (part_precision + part_recall)
            else:
                part_f1 = 0

            if (joint_precision + joint_recall) > 0:
                joint_f1 = 2 * (joint_precision * joint_recall) / (joint_precision + joint_recall)
            else:
                joint_f1 = 0

            
            M_joint_recall = len(best_match['M']) / best_match['num_gt_joints']
            MA_joint_recall = len(best_match['MA']) / best_match['num_gt_joints']
            MAO_joint_recall = len(best_match['MAO']) / best_match['num_gt_joints']

            if best_match['num_pred_joints'] > 0:
                M_joint_precision = len(best_match['M']) / best_match['num_pred_joints']
                MA_joint_precision = len(best_match['MA']) / best_match['num_pred_joints']
                MAO_joint_precision = len(best_match['MAO']) / best_match['num_pred_joints']
            else:
                M_joint_precision = 0
                MA_joint_precision = 0
                MAO_joint_precision = 0

            if (M_joint_precision + M_joint_recall) > 0:
                M_joint_f1 = 2 * (M_joint_precision * M_joint_recall) / (M_joint_precision + M_joint_recall)
            else:
                M_joint_f1 = 0

            if (MA_joint_precision + MA_joint_recall) > 0:
                MA_joint_f1 = 2 * (MA_joint_precision * MA_joint_recall) / (MA_joint_precision + MA_joint_recall)
            else:
                MA_joint_f1 = 0

            if (MAO_joint_precision + MAO_joint_recall) > 0:
                MAO_joint_f1 = 2 * (MAO_joint_precision * MAO_joint_recall) / (MAO_joint_precision + MAO_joint_recall)
            else:
                MAO_joint_f1 = 0

            eval_results['part_recall'].append(part_recall)
            eval_results['joint_recall'].append(joint_recall)
            eval_results['part_precision'].append(part_precision)
            eval_results['joint_precision'].append(joint_precision)
            eval_results['part_f1'].append(part_f1)
            eval_results['joint_f1'].append(joint_f1)

            eval_results['M_joint_recall'].append(M_joint_recall)
            eval_results['MA_joint_recall'].append(MA_joint_recall)
            eval_results['MAO_joint_recall'].append(MAO_joint_recall)

            eval_results['M_joint_precision'].append(M_joint_precision)
            eval_results['MA_joint_precision'].append(MA_joint_precision)
            eval_results['MAO_joint_precision'].append(MAO_joint_precision)

            eval_results['M_joint_f1'].append(M_joint_f1)
            eval_results['MA_joint_f1'].append(MA_joint_f1)
            eval_results['MAO_joint_f1'].append(MAO_joint_f1)

            eval_results['pred_part_sum'].append(best_match['num_pred_parts'])
            eval_results['gt_part_sum'].append(best_match['num_gt_parts'])
            eval_results['pred_joint_sum'].append(best_match['num_pred_joints'])
            eval_results['gt_joint_sum'].append(best_match['num_gt_joints'])
            eval_results['match_part_sum'].append(len(best_match['part_matches']))
            eval_results['match_joint_sum'].append(len(best_match['joint_matches']))

            eval_results['M_num'].append(len(best_match['M']))
            eval_results['MA_num'].append(len(best_match['MA']))
            eval_results['MAO_num'].append(len(best_match['MAO']))
            if len(best_match['iou']) > 0 and len(best_match['md']) > 0:
                names.append(best_match['object_id'])
        log.debug(names)
        log.debug(len(names))
        log.debug(len(best_matches))

        return eval_results

    @staticmethod
    def write_evaluation_results(eval_results, output_path):
        result_strs = []
        for key, val in eval_results.items():
            if key not in ['pred_part_sum', 'gt_part_sum', 'pred_joint_sum', 'gt_joint_sum', 'match_part_sum', 'match_joint_sum', 'M_num', 'MA_num', 'MAO_num']:
                tmp_str = f'mean {key}: {round(np.mean(val), 4)}'
                result_strs.append(tmp_str)
                log.info(tmp_str)
                std_err = np.std(val) / np.sqrt(np.size(val))

                tmp_str = f'std {key}: {round(std_err, 4)}'
                result_strs.append(tmp_str)
                log.info(tmp_str)

        # recall, precision, f1    
        pred_part_sum = np.sum(eval_results['pred_part_sum'])
        gt_part_sum = np.sum(eval_results['gt_part_sum'])
        pred_joint_sum = np.sum(eval_results['pred_joint_sum'])
        gt_joint_sum = np.sum(eval_results['gt_joint_sum'])
        match_part_sum = np.sum(eval_results['match_part_sum'])
        match_joint_sum = np.sum(eval_results['match_joint_sum'])

        M_num = np.sum(eval_results['M_num'])
        MA_num = np.sum(eval_results['MA_num'])
        MAO_num = np.sum(eval_results['MAO_num'])

        part_recall = match_part_sum / gt_part_sum
        joint_recall = match_joint_sum / gt_joint_sum
        part_precision = match_part_sum / pred_part_sum
        joint_precision = match_joint_sum / pred_joint_sum

        M_recall = M_num / gt_joint_sum
        MA_recall = MA_num / gt_joint_sum
        MAO_recall = MAO_num / gt_joint_sum

        M_precision = M_num / pred_joint_sum
        MA_precision = MA_num / pred_joint_sum
        MAO_precision = MAO_num / pred_joint_sum


        tmp_str = f'all part recall: {round(part_recall, 4)}'
        result_strs.append(tmp_str)
        log.info(tmp_str)
        tmp_str = f'all joint recall: {round(joint_recall, 4)}'
        result_strs.append(tmp_str)
        log.info(tmp_str)
        tmp_str = f'all part precision: {round(part_precision, 4)}'
        result_strs.append(tmp_str)
        log.info(tmp_str)
        tmp_str = f'all joint precision: {round(joint_precision, 4)}'
        result_strs.append(tmp_str)
        log.info(tmp_str)
        tmp_str = f'all part f1: {round(2*(part_precision * part_recall) / (part_precision + part_recall), 4)}'
        result_strs.append(tmp_str)
        log.info(tmp_str)
        tmp_str = f'all joint f1: {round(2*(joint_precision * joint_recall) / (joint_precision + joint_recall), 4)}'
        result_strs.append(tmp_str)
        log.info(tmp_str)

        tmp_str = f'M joint recall: {round(M_recall, 4)}'
        result_strs.append(tmp_str)
        log.info(tmp_str)
        tmp_str = f'M joint precision: {round(M_precision, 4)}'
        result_strs.append(tmp_str)
        log.info(tmp_str)
        tmp_str = f'M F1: {round(2*(M_precision * M_recall) / (M_precision + M_recall), 4)}'
        result_strs.append(tmp_str)
        log.info(tmp_str)

        tmp_str = f'MA joint recall: {round(MA_recall, 4)}'
        result_strs.append(tmp_str)
        log.info(tmp_str)
        tmp_str = f'MA joint precision: {round(MA_precision, 4)}'
        result_strs.append(tmp_str)
        log.info(tmp_str)
        tmp_str = f'MA F1: {round(2*(MA_precision * MA_recall) / (MA_precision + MA_recall), 4)}'
        result_strs.append(tmp_str)
        log.info(tmp_str)

        tmp_str = f'MAO joint recall: {round(MAO_recall, 4)}'
        result_strs.append(tmp_str)
        log.info(tmp_str)
        tmp_str = f'MAO joint precision: {round(MAO_precision, 4)}'
        result_strs.append(tmp_str)
        log.info(tmp_str)
        tmp_str = f'MAO F1: {round(2*(MAO_precision * MAO_recall) / (MAO_precision + MAO_recall), 4)}'
        result_strs.append(tmp_str)
        log.info(tmp_str)

        with open(output_path, 'w+') as fp:
            for line in result_strs:
                fp.write(f"{line}\n")


@hydra.main(config_path='configs', config_name='evaluate', version_base='1.1')
def main(cfg: DictConfig):
    OmegaConf.update(cfg, "paths.result_dir", io.to_abs_path(cfg.paths.result_dir, get_original_cwd()))
    nms_output_cfg = get_latest_nms_output_cfg(cfg.paths.postprocess)
    io.ensure_dir_exists(cfg.output_dir)

    evaluator = Evaluation(cfg)

    if cfg.eval_train:
        data_sets = ['train']
    else:
        data_sets = [cfg.test_split]
    for data_set in data_sets:
        if data_set == 'train':
            input_path = cfg.paths.preprocess.output.train
            output_path = nms_output_cfg.train
        elif data_set == 'val':
            input_path = cfg.paths.preprocess.output.val
            output_path = nms_output_cfg.val
        elif data_set == 'test':
            input_path = cfg.paths.preprocess.output.test
            output_path = nms_output_cfg.test

        eval_results = evaluator.evaluate(input_path, output_path, cfg.inst_pred_path, cfg.articulation_dataset_path)
        if cfg.articulation_dataset_path and cfg.eval_closed:
            eval_output_path = os.path.join(cfg.output_dir, f'{data_set}_eval_closed_results.txt')
        elif cfg.articulation_dataset_path and ~cfg.eval_closed:
            eval_output_path = os.path.join(cfg.output_dir, f'{data_set}_eval_opened_results.txt')
        elif cfg.inst_pred_path:
            eval_output_path = os.path.join(cfg.output_dir, f'{data_set}_eval_inst_pred_results.txt')
        else:
            eval_output_path = os.path.join(cfg.output_dir, f'{data_set}_eval_results.txt')

        evaluator.write_evaluation_results(eval_results, eval_output_path)

if __name__ == '__main__':
    start = time()
    main()
    end = time()

    duration_time = utils.duration_in_hours(end - start)
    log.info(f'Evaluation: Total time duration {duration_time}')
