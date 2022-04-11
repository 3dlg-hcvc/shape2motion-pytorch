from concurrent.futures import process
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

from tools.utils import io
from tools.utils.constant import JointType
from tools.visualizations import Visualizer


log = logging.getLogger('post_stage1')

class PostStage1Impl:
    def __init__(self, cfg):
        self.cfg = cfg
        self.anchor_pts_threshold = cfg.anchor_pts_threshold
        self.simmat_threshold = cfg.simmat_threshold
        self.part_proposal_threshold = cfg.part_proposal_threshold
        self.top_k_score_threshold = cfg.top_k_score_threshold
        self.top_score_threshold = cfg.top_score_threshold
        self.move_angle_param = eval(cfg.move_angle_param)
        self.move_trans_param = cfg.move_trans_param
        self.overlap_threshold = cfg.overlap_threshold
        self.confidence_threshold = cfg.confidence_threshold
        self.min_points_threshold = cfg.min_points_threshold

    def compute_part_proposal_score(self, gt_part_proposals, pred_part_proposals):
        scores = np.zeros((gt_part_proposals.shape[0], pred_part_proposals.shape[0]))
        for i, gt_part_proposal in enumerate(gt_part_proposals):
            gt_part_proposal = gt_part_proposal.astype(float)
            gt_part_proposal = np.tile(gt_part_proposal, (pred_part_proposals.shape[0], 1))
            union = (gt_part_proposal + pred_part_proposals).astype(bool)
            inter = (gt_part_proposal * pred_part_proposals).astype(bool)
            score = np.sum(inter, axis=1) / np.sum(union, axis=1)
            scores[i] = score
        return scores

    def compute_motion_proposal_score(self, gt_score_idx, gt_part_proposals, gt_move_pts_list, pred_move_pts_list, turn_idx, pred_anchor_pts_idx):
        def compute_motion_proposal_score_step(part_mask, gt_move_part_pts, pred_move_pts_list):
            scores = np.zeros(len(pred_move_pts_list))
            for i, pred_move_pts in enumerate(pred_move_pts_list):
                pred_move_part_pts = pred_move_pts[part_mask, :]
                dist = LA.norm(gt_move_part_pts - pred_move_part_pts, axis=1)
                if dist.size == 0:
                    continue
                mean_dist = np.mean(dist)
                score = -2.0 / (1.0 + np.exp(-4.0 * mean_dist)) + 2.0
                scores[i] = score
            return scores    

        num_points = gt_move_pts_list[0].shape[0]
        motion_scores_all = np.zeros((gt_score_idx.shape[0], num_points))
        for i, gt_idx in enumerate(gt_score_idx):
            part_mask = np.where(gt_part_proposals[gt_idx, :] == 1)[0]
            gt_move_pts = gt_move_pts_list[gt_idx]
            gt_move_part_pts = gt_move_pts[part_mask, :]
            scores_step = compute_motion_proposal_score_step(part_mask, gt_move_part_pts, pred_move_pts_list)
            motion_scores = np.zeros(num_points)
            motion_scores[pred_anchor_pts_idx] = scores_step

            have_turn = np.where(turn_idx == gt_idx+1)[0].size > 0
            if have_turn and len(gt_move_pts_list) > gt_idx+1:
                gt_move_pts_2 = gt_move_pts_list[gt_idx+1]
                gt_move_part_pts_2 = gt_move_pts_2[part_mask, :]
                scores_step_2 = compute_motion_proposal_score_step(part_mask, gt_move_part_pts_2, pred_move_pts_list)
                motion_scores_2 = np.zeros(num_points)
                motion_scores_2[pred_anchor_pts_idx] = scores_step_2
                motion_scores = np.maximum(motion_scores, motion_scores_2)

            motion_scores_all[i] = motion_scores
        return motion_scores_all

    def move_pts_with_joints(self, pts, joint_origins, joint_directions, joint_types):
        move_pts_list = []
        for i, joint_type in enumerate(joint_types):
            if joint_type == JointType.ROT.value:
                move_pts = self.rot3d(pts, joint_origins[i], joint_directions[i])
            elif joint_type == JointType.TRANS.value:
                move_pts = self.trans3d(pts, joint_directions[i])
            elif joint_type == JointType.BOTH.value:
                move_pts = self.rot3d(pts, joint_origins[i], joint_directions[i])
                move_pts = self.trans3d(move_pts, joint_directions[i])
            else:
                raise NotImplementedError(f'No implementation for the joint type value {joint_type}')
            move_pts_list.append(move_pts)
        return move_pts_list

    def rot3d(self, pts, joint_origin, joint_direction):
        joint_direction = joint_direction / LA.norm(joint_direction)
        rot_mat = R.from_rotvec(self.move_angle_param * joint_direction).as_matrix()
        rot_pts = np.dot(pts - joint_origin, rot_mat.transpose()) + joint_origin
        return rot_pts

    def trans3d(self, pts, joint_direction):
        diag_length = LA.norm(np.amax(pts, axis=0) - np.amin(pts, axis=0))
        move_trans = self.move_trans_param * diag_length
        joint_direction = joint_direction / LA.norm(joint_direction)
        shift_vec = joint_direction * move_trans
        trans_pts = pts + shift_vec
        return trans_pts

    def process_train(self, idx, data):
        instance_name = data.instance_name
        input_pts = data.input_pts
        input_xyz = input_pts[:, :3]

        pred_anchor_pts = data.pred_anchor_pts
        pred_joint_type = data.pred_joint_type
        pred_joint_direction_cat = data.pred_joint_direction_cat
        pred_joint_direction_reg = data.pred_joint_direction_reg
        pred_joint_origin_reg = data.pred_joint_origin_reg
        pred_simmat = data.pred_simmat

        joint_all_directions = data.joint_all_directions
        gt_proposals = data.gt_proposals
        gt_joints = data.gt_joints

        # process predicted joint types
        pred_joint_type = pred_joint_type[1:, :]
        # rotaiton: 0, translation: 1
        pred_joint_type = np.argmax(pred_joint_type, axis=0) + 1

        # process predicted joint direction category, discard the index 0
        pred_joint_direction_cat = pred_joint_direction_cat[1:, :]
        pred_joint_direction_cat = np.argmax(pred_joint_direction_cat, axis=0)

        # process predicted anchor points
        pred_anchor_pts_exp = np.exp(pred_anchor_pts)
        # softmax
        pred_anchor_pts_softmax = pred_anchor_pts_exp / np.sum(pred_anchor_pts_exp, axis=0)
        pred_anchor_pts_idx = np.where(pred_anchor_pts_softmax[1, :] > self.anchor_pts_threshold)[0]

        if pred_anchor_pts_idx.size == 0:
            log.warning(f'{instance_name} stage 1 prediction failed: bad anchor points prediction')
            return

        # select joint parameters from anchor points
        pred_joint_type = pred_joint_type[pred_anchor_pts_idx]
        pred_joint_direction_cat = pred_joint_direction_cat[pred_anchor_pts_idx]
        pred_joint_direction_reg = pred_joint_direction_reg[pred_anchor_pts_idx, :]
        pred_joint_origin_reg = pred_joint_origin_reg[pred_anchor_pts_idx, :]
        # joint origin = anchor xyz + predicted joint origin reg
        pred_joint_origin = input_xyz[pred_anchor_pts_idx, :] + pred_joint_origin_reg
        # joint direction = base joint direction in 14 categories + predicted joint direction reg
        pred_joint_direction = joint_all_directions[pred_joint_direction_cat, :] + pred_joint_direction_reg

        # process similarity matrix
        s_mat = np.zeros_like(pred_simmat)
        s_mat[pred_simmat < self.simmat_threshold] = 1
        pred_part_proposals = np.unique(s_mat, axis=0)
        # discard base part proposals
        gt_part_proposals = gt_proposals[1:, :]
        scores = self.compute_part_proposal_score(gt_part_proposals, pred_part_proposals)
        gt_score_idx = np.argmax(scores, axis=0)
        gt_score_max = scores[gt_score_idx, np.arange(scores.shape[1])]
        # filter low scores
        pred_part_proposals = pred_part_proposals[gt_score_max > self.part_proposal_threshold]
        gt_score_idx = gt_score_idx[gt_score_max > self.part_proposal_threshold]
        gt_score_max = gt_score_max[gt_score_max > self.part_proposal_threshold]

        gt_score_idx_unique = np.unique(gt_score_idx)
        part_proposal_idx = []
        for gt_part_idx in gt_score_idx_unique:
            tmp_index = np.where(gt_score_idx == gt_part_idx)[0]
            tmp_score_max = gt_score_max[tmp_index]
            # tmp_score_idx = gt_score_idx[tmp_index]
            # sort in descending order
            top_score_idx = tmp_score_max.argsort()[::-1][:self.top_k_score_threshold]
            top_score_max = tmp_score_max[top_score_idx]
            high_score_idx = top_score_idx[top_score_max > self.top_score_threshold]
            if high_score_idx.size == 0:
                # now score is above threshold
                continue
            tmp_proposal_idx = tmp_index[high_score_idx]
            part_proposal_idx.append(tmp_proposal_idx)

        if len(part_proposal_idx) == 0:
            # no valid part proposal
            log.warning(f'{instance_name} stage 1 prediction failed: bad part proposal')
            return
        
        part_proposal_idx = np.concatenate(part_proposal_idx)
        gt_score_idx = gt_score_idx[part_proposal_idx]
        pred_part_proposals = pred_part_proposals[part_proposal_idx, :]
        gt_move_pts_list = self.move_pts_with_joints(input_xyz, gt_joints[:, :3], gt_joints[:, 3:6], gt_joints[:, 6])
        pred_move_pts_list = self.move_pts_with_joints(input_xyz, pred_joint_origin, pred_joint_direction, pred_joint_type)

        assert gt_score_idx.shape[0] > 0, 'Zero proposal'
        
        # where the object moves as a whole
        turn_idx = np.where(np.sum(gt_part_proposals, axis=1) == 0)[0]

        motion_scores = self.compute_motion_proposal_score(gt_score_idx, gt_part_proposals, gt_move_pts_list, pred_move_pts_list, turn_idx, pred_anchor_pts_idx)

        num_points = gt_move_pts_list[0].shape[0]
        # stage1 predicted results and ground truth
        pred_anchor_mask = np.zeros(num_points)
        pred_anchor_mask[pred_anchor_pts_idx] = 1
        pred_motions = np.zeros((num_points, 7))
        pred_motions[pred_anchor_pts_idx, :] = np.concatenate((pred_joint_origin, pred_joint_direction, pred_joint_type.reshape(-1, 1)), axis=1)
        gt_part_proposals = gt_proposals[1:, :]
        gt_part_proposals = gt_part_proposals[gt_score_idx, :]
        gt_motions = gt_joints[gt_score_idx]

        assert gt_part_proposals.shape[0] == pred_part_proposals.shape[0], 'Mismatch in prediction and gt part proposals'
        
        output_data = {}
        output_data['instance_name'] = instance_name
        output_data['input_pts'] = input_pts
        output_data['motion_scores'] = motion_scores
        output_data['pred_anchor_mask'] = pred_anchor_mask
        output_data['pred_part_proposals'] = pred_part_proposals
        output_data['pred_motions'] = pred_motions
        output_data['gt_part_proposals'] = gt_part_proposals
        output_data['gt_motions'] = gt_motions
        return output_data

    def process_test(self, idx, data):
        instance_name = data.instance_name
        input_pts = data.input_pts
        input_xyz = input_pts[:, :3]

        pred_anchor_pts = data.pred_anchor_pts
        pred_joint_type = data.pred_joint_type
        pred_joint_direction_cat = data.pred_joint_direction_cat
        pred_joint_direction_reg = data.pred_joint_direction_reg
        pred_joint_origin_reg = data.pred_joint_origin_reg
        pred_simmat = data.pred_simmat
        pred_confidences = data.pred_confidence.flatten()

        joint_all_directions = data.joint_all_directions
        gt_proposals = data.gt_proposals
        gt_joints = data.gt_joints

        # process predicted joint types
        pred_joint_type = pred_joint_type[1:, :]
        # rotaiton: 0, translation: 1
        pred_joint_type = np.argmax(pred_joint_type, axis=0) + 1

        # process predicted joint direction category, discard the index 0
        pred_joint_direction_cat = pred_joint_direction_cat[1:, :]
        pred_joint_direction_cat = np.argmax(pred_joint_direction_cat, axis=0)

        # process predicted anchor points
        pred_anchor_pts_exp = np.exp(pred_anchor_pts)
        # softmax
        pred_anchor_pts_softmax = pred_anchor_pts_exp / np.sum(pred_anchor_pts_exp, axis=0)
        pred_anchor_pts_idx = np.where(pred_anchor_pts_softmax[1, :] > self.anchor_pts_threshold)[0]

        if pred_anchor_pts_idx.size == 0:
            log.warning(f'{instance_name} stage 1 prediction failed: bad anchor points prediction')
            return

        # select joint parameters from anchor points
        pred_joint_type = pred_joint_type[pred_anchor_pts_idx]
        pred_joint_direction_cat = pred_joint_direction_cat[pred_anchor_pts_idx]
        pred_joint_direction_reg = pred_joint_direction_reg[pred_anchor_pts_idx, :]
        pred_joint_origin_reg = pred_joint_origin_reg[pred_anchor_pts_idx, :]
        # joint origin = anchor xyz + predicted joint origin reg
        pred_joint_origin = input_xyz[pred_anchor_pts_idx, :] + pred_joint_origin_reg
        # joint direction = base joint direction in 14 categories + predicted joint direction reg
        pred_joint_direction = joint_all_directions[pred_joint_direction_cat, :] + pred_joint_direction_reg

        # process similarity matrix
        s_mat = np.zeros_like(pred_simmat)
        s_mat[pred_simmat < self.simmat_threshold] = 1
        valid_pts_mask = np.sum(s_mat, axis=1) > self.min_points_threshold
        s_mat = s_mat[valid_pts_mask, :]
        pred_confidences = pred_confidences[valid_pts_mask]
        pred_part_proposals, proposal_indices = np.unique(s_mat, return_index=True, axis=0)
        pred_confidences = pred_confidences[proposal_indices]

        good_confidence_idx = np.where(pred_confidences > self.confidence_threshold)[0]
        pred_confidences = pred_confidences[good_confidence_idx]
        pred_part_proposals = pred_part_proposals[good_confidence_idx, :]

        part_pick = self.nms(pred_part_proposals, pred_confidences)
        pred_part_proposals = pred_part_proposals[part_pick, :]

        num_points = input_xyz.shape[0]
        # stage1 predicted results and ground truth
        pred_anchor_mask = np.zeros(num_points)
        pred_anchor_mask[pred_anchor_pts_idx] = 1

        pred_motions = np.zeros((num_points, 7))
        pred_motions[pred_anchor_pts_idx, :] = np.concatenate((pred_joint_origin, pred_joint_direction, pred_joint_type.reshape(-1, 1)), axis=1)

        gt_move_pts_list = self.move_pts_with_joints(input_xyz, gt_joints[:, :3], gt_joints[:, 3:6], gt_joints[:, 6])
        pred_move_pts_list = self.move_pts_with_joints(input_xyz, pred_joint_origin, pred_joint_direction, pred_joint_type)

        gt_part_proposals = gt_proposals[1:, :]
        turn_idx = np.where(np.sum(gt_part_proposals, axis=1) == 0)[0]
        scores = self.compute_part_proposal_score(gt_part_proposals, pred_part_proposals)
        gt_score_idx = np.argmax(scores, axis=0)
        motion_scores = self.compute_motion_proposal_score(gt_score_idx, gt_part_proposals, gt_move_pts_list, pred_move_pts_list, turn_idx, pred_anchor_pts_idx)
        gt_part_proposals = gt_part_proposals[gt_score_idx, :]
        gt_motions = gt_joints[gt_score_idx]

        output_data = {}
        output_data['instance_name'] = instance_name
        output_data['input_pts'] = input_pts
        output_data['motion_scores'] = motion_scores
        output_data['pred_anchor_mask'] = pred_anchor_mask
        output_data['pred_part_proposals'] = pred_part_proposals
        output_data['pred_motions'] = pred_motions
        output_data['gt_part_proposals'] = gt_part_proposals
        output_data['gt_motions'] = gt_motions
        return output_data

    def nms(self, masks, scores):
        I = np.argsort(scores)
        pick = []
        while (I.size!=0):
            last = I.size
            i = I[-1]
            pick.append(i)

            this_mask = np.tile(masks[i], (len(I[:last-1]), 1))

            inter = np.logical_and(this_mask, masks[I[:last-1]])
            o = inter / (np.logical_or(this_mask, masks[I[:last-1]]) + 1.0e-9)
            I = np.delete(I, np.concatenate(([last-1], np.where(o > self.overlap_threshold)[0])))

        return pick

    def __call__(self, idx, data):
        data = SimpleNamespace(**data)
        if data.data_set == 'train':
            return self.process_train(idx, data)
        else:
            return self.process_test(idx, data)
        


class PostStage1:
    def __init__(self, cfg):
        self.cfg = cfg
        self.debug = self.cfg.debug
        self.num_workers = self.cfg.num_workers

        self.gt_h5 = None
        self.df = pd.DataFrame()
        self.output_h5 = None
        self.data_set = 'train'

    def set_datapath(self, data_path, output_path, data_set):
        assert io.is_non_zero_file(data_path), OSError(f'Cannot find file {data_path}')
        self.gt_h5 = h5py.File(data_path, 'r')
        io.ensure_dir_exists(os.path.dirname(output_path))
        self.output_h5 = h5py.File(output_path, 'w')
        self.data_set = data_set

    def process(self, pred, input_pts, gt, id):
        input_pts = input_pts.detach().cpu().numpy()
        
        pred_anchor_pts_batch = pred['anchor_pts'].detach().cpu().numpy()
        pred_joint_direction_cat_batch = pred['joint_direction_cat'].detach().cpu().numpy()
        pred_joint_direction_reg_batch = pred['joint_direction_reg'].detach().cpu().numpy()
        pred_joint_origin_reg_batch = pred['joint_origin_reg'].detach().cpu().numpy()
        pred_joint_type_batch = pred['joint_type'].detach().cpu().numpy()
        pred_simmat_batch = pred['simmat'].detach().cpu().numpy()
        pred_confidence_batch = pred['confidence'].detach().cpu().numpy()
        pred_simmat_batch = (pred_simmat_batch <= 255) * pred_simmat_batch + (pred_simmat_batch > 255) * 255
        pred_simmat_batch = pred_simmat_batch.astype(np.uint8)

        batch_size = pred_anchor_pts_batch.shape[0]

        # prepare input data for the stage1 postprocessing
        stage1_data = []
        for b in range(batch_size):
            tmp_data = {}
            instance_name = id[b]
            tmp_data['input_pts'] = input_pts[b]
            tmp_data['data_set'] = self.data_set
            tmp_data['instance_name'] = instance_name
            tmp_data['pred_anchor_pts'] = pred_anchor_pts_batch[b]
            tmp_data['pred_joint_direction_cat'] = pred_joint_direction_cat_batch[b]
            tmp_data['pred_joint_direction_reg'] = pred_joint_direction_reg_batch[b]
            tmp_data['pred_joint_origin_reg'] = pred_joint_origin_reg_batch[b]
            tmp_data['pred_joint_type'] = pred_joint_type_batch[b]
            tmp_data['pred_simmat'] = pred_simmat_batch[b]
            tmp_data['pred_confidence'] = pred_confidence_batch[b]

            gt_instance = self.gt_h5[instance_name]
            tmp_data['joint_all_directions'] = gt_instance['joint_all_directions'][:]

            tmp_data['gt_joints'] = gt_instance['gt_joints'][:]
            tmp_data['gt_proposals'] = gt_instance['gt_proposals'][:]

            stage1_data.append(tmp_data)

        pool = Pool(processes=self.num_workers)
        proc_impl = PostStage1Impl(self.cfg)
        jobs = [pool.apply_async(proc_impl, args=(i,data,)) for i, data in enumerate(stage1_data)]
        pool.close()
        pool.join()
        batch_output = [job.get() for job in jobs]

        # batch_output = []
        # for i, data in enumerate(stage1_data):
        #     batch_output.append(proc_impl(i, data))

        for output_data in batch_output:
            if output_data is None:
                continue
            instance_name = output_data['instance_name']
            input_pts = output_data['input_pts']
            pred_anchor_mask = output_data['pred_anchor_mask']
            pred_part_proposals = output_data['pred_part_proposals']
            pred_motions = output_data['pred_motions']
            motion_scores = output_data['motion_scores']
            gt_part_proposals = output_data['gt_part_proposals']
            gt_motions = output_data['gt_motions']

            h5instance = self.output_h5.require_group(instance_name)
            h5instance.create_dataset('input_pts', shape=input_pts.shape, data=input_pts, compression='gzip')
            h5instance.create_dataset('pred_anchor_mask', shape=pred_anchor_mask.shape, data=pred_anchor_mask, compression='gzip')
            h5instance.create_dataset('pred_part_proposals', shape=pred_part_proposals.shape, data=pred_part_proposals, compression='gzip')
            h5instance.create_dataset('pred_motions', shape=pred_motions.shape, data=pred_motions, compression='gzip')
            h5instance.create_dataset('motion_scores', shape=motion_scores.shape, data=motion_scores, compression='gzip')
            h5instance.create_dataset('gt_part_proposals', shape=gt_part_proposals.shape, data=gt_part_proposals, compression='gzip')
            h5instance.create_dataset('gt_motions', shape=gt_motions.shape, data=gt_motions, compression='gzip')

            if self.debug:
                gt_cfg = {}
                gt_cfg['part_proposals'] = gt_part_proposals
                gt_cfg['motions'] = gt_motions
                gt_cfg = SimpleNamespace(**gt_cfg)

                pred_cfg = {}
                pred_cfg['part_proposals'] = pred_part_proposals
                pred_cfg['motions'] = pred_motions
                pred_cfg['scores'] = motion_scores
                pred_cfg['anchor_mask'] = pred_anchor_mask
                pred_cfg = SimpleNamespace(**pred_cfg)

                viz = Visualizer(input_pts[:, :3])
                viz.view_stage2_input(gt_cfg, pred_cfg)
    
    def stop(self):
        self.output_h5.close()
