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

    def compute_dist(self, pred_origin, gt_joint):
        q1 = gt_joint[:3]
        q2 = gt_joint[:3] + gt_joint[3:6]
        vec1 = q2 - q1
        vec2 = pred_origin - q1
        dist = LA.norm(np.cross(vec1, vec2) / (LA.norm(vec1) * LA.norm(vec2)))
        return dist

    def evaluate(self, gt_h5file, pred_h5file):
        gt_h5 = h5py.File(gt_h5file, 'r')
        pred_h5 = h5py.File(pred_h5file, 'r')
        
        best_matches = []
        for object_name in pred_h5.keys():

            best_match = {
                'object_name': object_name,
                'iou': [],
                'md': [],
                'oe': [],
                'ta': [],
                'part_matches': {},
                'joint_matches': {},
            }
            
            gt_object = gt_h5[object_name]
            gt_proposals = gt_object['gt_proposals'][:][1:, :].astype(bool)
            gt_joints = gt_object['gt_joints'][:]

            turn_idx = np.where(np.sum(gt_proposals, axis=1) == 0)[0]

            pred_object = pred_h5[object_name]
            pred_part_proposals = pred_object['pred_part_proposal'][:]
            pred_joints = pred_object['pred_joints'][:]
            pred_scores = pred_object['pred_scores'][:]
            pred_joints_map = pred_object['pred_joints_map'][:]
            
            best_parts = np.ones(gt_proposals.shape[0]) * -1.0
            best_ious = np.ones(gt_proposals.shape[0]) * -1.0
            for part_idx in range(gt_proposals.shape[0]):
                gt_proposal = gt_proposals[part_idx, :]
                best_part = -1
                best_iou = -1
                for pred_part_idx in range(np.unique(pred_part_proposals).shape[0]-1):
                    if pred_part_idx in best_parts:
                        continue
                    pred_part_proposal = pred_part_proposals == (pred_part_idx+1)
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
                have_turn = np.where(turn_idx == part_idx+1)[0].size > 0
                pred_joints_idx = np.where(pred_joints_map == best_part)[0]
                pred_part_joints_scores = pred_scores[pred_joints_idx]
                pred_part_joints_sorted_idx = np.argsort(pred_part_joints_scores)[::-1]
                pred_part_joints_sorted_idx = pred_joints_idx[pred_part_joints_sorted_idx]
                pred_part_joints = pred_joints[pred_part_joints_sorted_idx, :]
                gt_joint = gt_joints[part_idx]
                selected_joint = None
                for j, joint in enumerate(pred_part_joints):
                    if not np.any(joint):
                        continue
                    md = None
                    oe = None
                    ta = None
                    if part_idx not in best_match['joint_matches'].keys():
                        md = self.compute_dist(joint[:3], gt_joint)
                        oe = np.arccos(np.clip(np.dot(joint[3:6], gt_joint[3:6]) / (LA.norm(joint[3:6]) * LA.norm(gt_joint[3:6])), -1, 1))
                        ta = joint[6] == gt_joint[6]
                        selected_joint = gt_joint
                    if have_turn:
                        if part_idx + 1 not in best_match['joint_matches'].keys():
                            gt_joint2 = gt_joints[part_idx+1]
                            md2 = self.compute_dist(joint[:3], gt_joint2)
                            oe2 = np.arccos(np.clip(np.dot(joint[3:6], gt_joint2[3:6]) / (LA.norm(joint[3:6]) * LA.norm(gt_joint2[3:6])), -1, 1))
                            ta2 = joint[6] == gt_joint2[6]
                            if oe is not None and oe2 < oe:
                                best_match['joint_matches'][part_idx+1] = pred_part_joints_sorted_idx[j]
                                md = np.minimum(md, md2)
                                oe = np.minimum(oe, oe2)
                                ta = np.maximum(ta, ta2)
                                selected_joint = gt_joint2
                            elif oe is not None and oe2 >= oe:
                                best_match['joint_matches'][part_idx] = pred_part_joints_sorted_idx[j]
                                md = np.minimum(md, md2)
                                oe = np.minimum(oe, oe2)
                                ta = np.maximum(ta, ta2)
                                selected_joint = gt_joint
                            else:
                                md = md2
                                oe = oe2
                                ta = ta2
                                selected_joint = gt_joint2
                    if md is not None:
                        best_match['md'].append(md)
                        best_match['oe'].append(oe)
                        best_match['ta'].append(ta)

                        if self.cfg.debug:
                            gt_cfg = {}
                            gt_cfg['part_proposal'] = gt_proposal
                            gt_cfg['joint'] = selected_joint
                            gt_cfg = SimpleNamespace(**gt_cfg)

                            pred_cfg = {}
                            pred_cfg['part_proposal'] = pred_part_proposals == (best_part+1)
                            pred_cfg['joint'] = joint
                            pred_cfg = SimpleNamespace(**pred_cfg)

                            input_xyz = gt_object['input_pts'][:][:, :3]
                            viz = Visualizer(input_xyz)
                            viz.view_evaluation_result(gt_cfg, pred_cfg)

            best_matches.append(best_match)

        eval_results = {
            'iou': [],
            'md': [],
            'oe': [],
            'ta': [],
        }
        for best_match in best_matches:
            eval_results['iou'] += best_match['iou']
            eval_results['md'] += best_match['md']
            eval_results['oe'] += best_match['oe']
            eval_results['ta'] += best_match['ta']
        for key, val in eval_results.items():
            log.info(f'mean {key}: {np.mean(val)}')


@hydra.main(config_path='configs', config_name='evaluate')
def main(cfg: DictConfig):
    OmegaConf.update(cfg, "paths.result_dir", io.to_abs_path(cfg.paths.result_dir, get_original_cwd()))
    nms_output_cfg = get_latest_nms_output_cfg(cfg.paths.postprocess)
    
    evaluator = Evaluation(cfg)

    # data_sets = ['train', cfg.test_split]
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
        
        evaluator.evaluate(input_path, output_path)

if __name__ == '__main__':
    start = time()
    main()
    end = time()

    duration_time = utils.duration_in_hours(end - start)
    log.info(f'Evaluation: Total time duration {duration_time}')