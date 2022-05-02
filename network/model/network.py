from http.client import NOT_IMPLEMENTED
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.model.backbone import PointNet2
from network.model import loss
from tools.utils.constant import Stage

from time import time


class Shape2Motion(nn.Module):
    def __init__(self, stage, device, num_points, num_channels):
        super().__init__()
        self.stage = Stage[stage] if isinstance(stage, str) else stage
        self.device = device
        self.epsilon = 1e-9

        # Define the shared PN++
        self.backbone = PointNet2(num_channels)

        if self.stage == Stage.stage2:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if self.stage == Stage.stage1 or self.stage == Stage.stage2:
            # motion proposal branch
            self.motion_feat = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(True)
            )
            # similarity matrix branch
            self.simmat_feat = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(True)
            )

            if self.stage == Stage.stage2:
                for param in self.motion_feat.parameters():
                    param.requires_grad = False

                for param in self.simmat_feat.parameters():
                    param.requires_grad = False

        if self.stage == Stage.stage1:
            # task1: key_point
            self.feat1 = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(True)
            )
            self.anchor_pts_layer = nn.Conv1d(128, 2, kernel_size=1, padding=0)

            # task2_1: joint_direction_category
            self.feat2_1 = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(True)
            )
            self.joint_direction_cat_layer = nn.Conv1d(128, 15, kernel_size=1, padding=0)

            # task2_2: joint_direction_regression
            self.feat2_2 = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(True)
            )
            self.joint_direction_reg_layer = nn.Conv1d(128, 3, kernel_size=1, padding=0)

            # task_3: joint_origin_regression
            self.feat3 = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(True)
            )
            self.joint_origin_reg_layer = nn.Conv1d(128, 3, kernel_size=1, padding=0)

            # task_4: joint_type
            self.feat4 = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(True)
            )
            self.joint_type_layer = nn.Conv1d(128, 4, kernel_size=1, padding=0)

            # task_5: similar matrix
            self.feat5 = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(True)
            )

            # task_6: confidence
            self.feat6 = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(True)
            )
            self.confidence_layer = nn.Conv1d(128, 1, kernel_size=1, padding=0)

        elif self.stage == Stage.stage2:
            self.motion_feat_1 = nn.Sequential(
                nn.Conv1d(128, 512, kernel_size=1, padding=0),
                nn.BatchNorm1d(512),
                nn.ReLU(True)
            )

            self.simmat_feat_1 = nn.Sequential(
                nn.Conv1d(128, 512, kernel_size=1, padding=0),
                nn.BatchNorm1d(512),
                nn.ReLU(True)
            )

            self.feat1 = nn.Sequential(
                nn.Conv1d(1024, 1024, kernel_size=1, padding=0),
                nn.BatchNorm1d(1024),
                nn.ReLU(True)
            )

            self.feat2 = nn.Sequential(
                nn.Conv1d(1024, 512, kernel_size=1, padding=0),
                nn.BatchNorm1d(512),
                nn.ReLU(True)
            )

            self.feat3 = nn.Sequential(
                nn.Conv1d(512, 256, kernel_size=1, padding=0),
                nn.BatchNorm1d(256),
                nn.ReLU(True)
            )

            self.motion_score_layer = nn.Conv1d(256, 1, kernel_size=1, padding=0)

        elif self.stage == Stage.stage3:
            self.dynamic_backbone = PointNet2(num_channels)

            # static branch
            self.static_feat = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(True)
            )
            # dynamic branch
            self.dynamic_feat = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(True)
            )

            self.proposal_feat = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(True)
            )
            self.proposal_layer = nn.Sequential(
                nn.Conv1d(128, 2, kernel_size=1, padding=0),
                nn.BatchNorm1d(2),
                nn.ReLU(True)
            )

            self.regression_feat = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(True)
            )
            self.regression_layer = nn.Sequential(
                nn.Conv1d(128, 6, kernel_size=num_points, padding=0),
                nn.BatchNorm1d(6),
                nn.ReLU(True)
            )
        else:
            raise NotImplementedError(f'No implementation for the stage {self.stage.value}')

    def forward(self, input, gt=None):
        batch_size = input.size(dim=0)

        features = self.backbone(input)
        if self.stage == Stage.stage3:
            moved_pcds = gt['moved_pcds']
            dynamic_features_1 = self.dynamic_backbone(moved_pcds[:, 0, :, :])
            dynamic_features_1 = torch.unsqueeze(dynamic_features_1, 1)
            dynamic_features_2 = self.dynamic_backbone(moved_pcds[:, 1, :, :])
            dynamic_features_2 = torch.unsqueeze(dynamic_features_2, 1)
            dynamic_features_3 = self.dynamic_backbone(moved_pcds[:, 2, :, :])
            dynamic_features_3 = torch.unsqueeze(dynamic_features_3, 1)
            dynamic_features = torch.cat((dynamic_features_1, dynamic_features_2, dynamic_features_3), axis=1)
            static_features = torch.unsqueeze(features, 1)

        if self.stage == Stage.stage1 or self.stage == Stage.stage2:
            motion_feat = self.motion_feat(features)
            simmat_feat = self.simmat_feat(features)

        if self.stage == Stage.stage1:
            feat1 = self.feat1(motion_feat)
            pred_anchor_pts = self.anchor_pts_layer(feat1)

            feat2_1 = self.feat2_1(motion_feat)
            pred_joint_direction_cat = self.joint_direction_cat_layer(feat2_1)

            feat2_2 = self.feat2_2(motion_feat)
            pred_joint_direction_reg = self.joint_direction_reg_layer(feat2_2).transpose(1, 2)

            feat3 = self.feat3(motion_feat)
            pred_joint_origin_reg = self.joint_origin_reg_layer(feat3).transpose(1, 2)

            feat4 = self.feat4(motion_feat)
            pred_joint_type = self.joint_type_layer(feat4)

            feat5 = self.feat5(simmat_feat).transpose(1, 2)
            r = torch.sum(feat5 * feat5, 2)
            r = torch.reshape(r, (batch_size, -1, 1))
            D = r - 2 * torch.matmul(feat5, feat5.transpose(1, 2)) + r.transpose(1, 2)
            pred_simmat = torch.clamp(10.0 * D, min=0)

            feat6 = self.feat6(simmat_feat)
            pred_confidence = self.confidence_layer(feat6).transpose(1, 2)
            pred_confidence = torch.sigmoid(pred_confidence)

            pred = {
                'anchor_pts': pred_anchor_pts,
                'joint_direction_cat': pred_joint_direction_cat,
                'joint_direction_reg': pred_joint_direction_reg,
                'joint_origin_reg': pred_joint_origin_reg,
                'joint_type': pred_joint_type,
                'simmat': pred_simmat,
                'confidence': pred_confidence,
            }
        elif self.stage == Stage.stage2:
            motion_feat_1 = self.motion_feat_1(motion_feat).transpose(1, 2)
            simmat_feat_1 = self.simmat_feat_1(simmat_feat).transpose(1, 2)

            part_proposal = gt['part_proposal']
            anchor_mask = gt['anchor_mask']

            num_points = part_proposal.size(dim=1)
            part_proposal = torch.unsqueeze(part_proposal, -1)
            part_proposal = part_proposal.repeat(1, 1, 512).float()
            simmat_feat_mul = simmat_feat_1 * part_proposal
            simmat_feat_max, _ = torch.max(simmat_feat_mul, axis=1)
            simmat_feat_max = torch.unsqueeze(simmat_feat_max, 1)
            simmat_feat_expand = simmat_feat_max.repeat(1, num_points, 1)
            all_feat = torch.cat((motion_feat_1, simmat_feat_expand), axis=2)

            anchor_mask = torch.unsqueeze(anchor_mask, -1)
            anchor_mask = anchor_mask.repeat(1, 1, 1024).float()
            anchor_feat = all_feat * anchor_mask

            anchor_feat_1 = self.feat1(anchor_feat.transpose(1, 2))
            anchor_feat_2 = self.feat2(anchor_feat_1)
            anchor_feat_3 = self.feat3(anchor_feat_2)

            pred_motion_scores = self.motion_score_layer(anchor_feat_3)
            pred_motion_scores = torch.sigmoid(pred_motion_scores.transpose(1, 2))
            pred_motion_scores = torch.squeeze(pred_motion_scores, -1)
            pred = {
                'motion_scores': pred_motion_scores
            }
        elif self.stage == Stage.stage3:
            all_feat = torch.cat((static_features, dynamic_features), axis=1)
            all_feat, _ = torch.max(all_feat, axis=1)

            proposal_feat = self.proposal_feat(all_feat)
            pred_proposal = self.proposal_layer(proposal_feat)
            regression_feat = self.regression_feat(all_feat)
            pred_regression = self.regression_layer(regression_feat)
            pred_regression = torch.squeeze(pred_regression, -1)

            pred = {
                'part_proposal': pred_proposal,
                'motion_regression': pred_regression,
            }
        return pred

    def losses(self, pred, gt):
        # The returned loss is a value
        if self.stage == Stage.stage1:
            mask = gt['anchor_pts'].float()

            # pred['pred_anchor_pts']: B*N*3K, gt['npcs_per_point']: B*N*3, gt_seg_onehot: B*N*K
            anchor_pts_loss, anchor_pts_recall, anchor_pts_accuracy = loss.compute_anchor_pts_loss(
                pred['anchor_pts'],
                gt['anchor_pts'].float(),
                mask=mask
            )

            joint_direction_cat_loss, joint_direction_cat_accuracy = loss.compute_joint_direction_cat_loss(
                pred['joint_direction_cat'],
                gt['joint_direction_cat'],
                mask=mask
            )

            joint_direction_reg_loss = loss.compute_joint_direction_reg_loss(
                pred['joint_direction_reg'],
                gt['joint_direction_reg'],
                mask=mask
            )

            joint_origin_reg_loss = loss.compute_joint_origin_reg_loss(
                pred['joint_origin_reg'],
                gt['joint_origin_reg'],
                mask=mask
            )

            joint_type_loss, joint_type_accuracy = loss.compute_joint_type_loss(
                pred['joint_type'],
                gt['joint_type'],
                mask=mask
            )

            sim_thresh = 80.0
            gt_simmat = gt['simmat'].float()
            tmp_neg_simmat = 1 - gt_simmat
            batch_size = gt_simmat.size(dim=0)
            mat_dim = gt_simmat.size(dim=1)
            eye_mat = torch.eye(gt_simmat.size(dim=1)).to(self.device)
            eye_mat = eye_mat.reshape((1, mat_dim, mat_dim))
            eye_batch = eye_mat.repeat(batch_size, 1, 1)
            neg_simmat = tmp_neg_simmat - eye_batch
            simmat_loss = loss.compute_simmat_loss(
                pred['simmat'],
                gt_simmat,
                neg_simmat,
                threshold=sim_thresh
            )

            confidence_loss = loss.compute_confidence_loss(
                pred['confidence'],
                pred['simmat'],
                gt_simmat,
                threshold=sim_thresh,
                epsilon=self.epsilon
            )

            loss_dict = {
                'anchor_pts_loss': anchor_pts_loss,
                'joint_direction_cat_loss': joint_direction_cat_loss,
                'joint_direction_reg_loss': joint_direction_reg_loss,
                'joint_origin_reg_loss': joint_origin_reg_loss,
                'joint_type_loss': joint_type_loss,
                'simmat_loss': simmat_loss,
                'confidence_loss': confidence_loss,
                'anchor_pts_recall': anchor_pts_recall,
                'anchor_pts_accuracy': anchor_pts_accuracy,
                'joint_direction_cat_accuracy': joint_direction_cat_accuracy,
                'joint_type_accuracy': joint_type_accuracy,
            }
        elif self.stage == Stage.stage2:
            anchor_mask = gt['anchor_mask'].float()
            gt_motion_scores = torch.unsqueeze(gt['motion_scores'], -1)
            pred_motion_scores = torch.unsqueeze(pred['motion_scores'], -1)

            motion_scores_loss = loss.compute_motion_scores_loss(
                pred_motion_scores,
                gt_motion_scores,
                anchor_mask,
                self.epsilon
            )

            loss_dict = {
                'motion_scores_loss': motion_scores_loss,
            }
        elif self.stage == Stage.stage3:
            part_proposal_loss, part_proposal_accuracy, iou = loss.compute_part_proposal_loss(
                pred['part_proposal'],
                gt['part_proposal'].float(),
                self.epsilon
            )
            motion_regression_loss = loss.compute_motion_regression_loss(
                pred['motion_regression'],
                gt['motion_regression'],
            )

            loss_dict = {
                'part_proposal_loss': part_proposal_loss,
                'part_proposal_accuracy': part_proposal_accuracy,
                'iou': iou,
                'motion_regression_loss': motion_regression_loss
            }

        return loss_dict
