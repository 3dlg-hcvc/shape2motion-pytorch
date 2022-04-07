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

import pdb

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

    def evaluate(self, gt_h5file, pred_h5file):
        gt_h5 = h5py.File(gt_h5file, 'r')
        pred_h5 = h5py.File(pred_h5file, 'r')

        best_matches = []
        for object_name in pred_h5.keys():
            best_match = {}
            best_match = {
                'object_name': object_name,
                'iou': [],
                'md': [],
                'oe': [],
                'ta': [],
                'matches': {},
            }
            gt_object = gt_h5[object_name]
            gt_proposals = gt_object['gt_proposals'][:][1:, :].astype(bool)
            gt_joints = gt_object['gt_joints'][:]

            best_match['gt_part_num'] = gt_proposals.shape[0]
            best_match['gt_joint_num'] = gt_joints.shape[0]

            pred_object = pred_h5[object_name]
            pred_part_proposals = pred_object['pred_part_proposal'][:]
            pred_joints = pred_object['pred_joints'][:]
            pred_scores = pred_object['pred_scores'][:]
            best_match['pred_part_num'] = np.unique(pred_part_proposals).shape[0] - 1
            best_match['pred_joint_num'] = pred_joints.shape[0]
            
            high_scores = np.argsort(pred_scores)[::-1]
            for idx in high_scores:
                tmp_pred_part_proposal = pred_part_proposals == (idx+1)
                tmp_pred_joints = pred_joints[idx]

                tmp_pred_part_proposal = np.tile(tmp_pred_part_proposal, (gt_proposals.shape[0], 1))
                inter = np.sum(np.logical_and(tmp_pred_part_proposal, gt_proposals), axis=1)
                outer = np.sum(np.logical_or(tmp_pred_part_proposal, gt_proposals), axis=1)
                iou = inter / outer

                good_part_candidates = np.where(iou > self.cfg.iou_threshold)[0]
                
                for chosen in best_match['matches'].values():
                    good_part_candidates = good_part_candidates[good_part_candidates != chosen]

                if len(good_part_candidates) > 0:
                    best_candidate = np.argmax(iou[good_part_candidates])
                    best_candidate = good_part_candidates[best_candidate]
                    best_match['matches'][idx] = best_candidate

                    # TODO: check how object with multiple joints will work here
                    selected_pred_joint = tmp_pred_joints
                    gt_joint = gt_joints[best_candidate]
                    md = self.compute_dist(selected_pred_joint[:3], gt_joint)
                    oe = np.arccos(np.clip(np.dot(selected_pred_joint[3:6], gt_joint[3:6]) / (LA.norm(selected_pred_joint[3:6]) * LA.norm(gt_joint[3:6])), -1, 1))
                    ta = selected_pred_joint[6] == gt_joint[6]
                    best_match['iou'].append(iou[best_candidate])
                    best_match['md'].append(md)
                    best_match['oe'].append(oe)
                    best_match['ta'].append(ta)
            if self.cfg.debug:
                tmp_pred_part_proposals = np.zeros_like(pred_part_proposals)
                for pred_idx, gt_idx in best_match['matches'].items():
                    tmp_pred_part_proposals[pred_part_proposals == (pred_idx+1)] = (gt_idx+1)
                tmp_gt_part_proposals = np.zeros_like(pred_part_proposals)
                for i, gt_proposal in enumerate(gt_proposals):
                    tmp_gt_part_proposals[gt_proposal] = i+1
                input_xyz = gt_object['input_pts'][:][:, :3]

                pdb.set_trace()

                gt_cfg = {}
                gt_cfg['part_proposal'] = tmp_gt_part_proposals
                gt_cfg['joints'] = gt_joints
                gt_cfg = SimpleNamespace(**gt_cfg)

                pred_cfg = {}
                pred_cfg['part_proposal'] = tmp_pred_part_proposals
                pred_cfg['joints'] = pred_joints
                pred_cfg = SimpleNamespace(**pred_cfg)

                viz = Visualizer(input_xyz)
                viz.view_evaluation_result(gt_cfg, pred_cfg)


            best_matches.append(best_match)

        eval_results = {
            'iou': [],
            'md': [],
            'oe': [],
            'ta': [],
            'part_recall': [],
            'joint_recall': [],
        }
        for best_match in best_matches:
            eval_results['iou'] += best_match['iou']
            eval_results['md'] += best_match['md']
            eval_results['oe'] += best_match['oe']
            eval_results['ta'] += best_match['ta']
            eval_results['part_recall'].append(best_match['pred_part_num'] / best_match['gt_part_num'])
            eval_results['joint_recall'].append(best_match['pred_joint_num'] / best_match['gt_joint_num'])
        for key, val in eval_results.items():
            log.info(f'mean {key}: {np.mean(val)}')
            
    def compute_dist(self, pred_origin, gt_joint):
        q1 = gt_joint[:3]
        q2 = gt_joint[:3] + gt_joint[3:6]
        vec1 = q2 - q1
        vec2 = pred_origin - q1
        dist = LA.norm(np.cross(vec1, vec2) / LA.norm(q2 - q1))
        return dist


@hydra.main(config_path='configs', config_name='evaluate')
def main(cfg: DictConfig):
    OmegaConf.update(cfg, "paths.result_dir", io.to_abs_path(cfg.paths.result_dir, get_original_cwd()))
    nms_output_cfg = get_latest_nms_output_cfg(cfg.paths.postprocess)

    evaluator = Evaluation(cfg)

    data_sets = ['train', cfg.test_split]
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
