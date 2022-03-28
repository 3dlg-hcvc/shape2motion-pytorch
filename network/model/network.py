from http.client import NOT_IMPLEMENTED
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.model.backbone import PointNet2
from network.model import loss
from network.utils import Stage


class Shape2Motion(nn.Module):
    def __init__(self, stage, device):
        super().__init__()
        self.stage = Stage[stage] if isinstance(stage, str) else stage
        self.device = device

        # Define the shared PN++
        self.backbone = PointNet2()

        if self.stage == Stage.stage1:
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
            
            #task1: key_point
            self.feat1 = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.ReLU(True)
            )
            self.anchor_pts_layer = nn.Conv1d(128, 2, kernel_size=1, padding=0)

            #task2_1: joint_direction_category
            self.feat2_1 = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.ReLU(True)
            )
            self.joint_direction_cat_layer = nn.Conv1d(128, 15, kernel_size=1, padding=0)

            #task2_2: joint_direction_regression
            self.feat2_2 = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.ReLU(True)
            )
            self.joint_direction_reg_layer = nn.Conv1d(128, 3, kernel_size=1, padding=0)

            #task_3: joint_origin_regression
            self.feat3 = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.ReLU(True)
            )
            self.joint_origin_reg_layer = nn.Conv1d(128, 3, kernel_size=1, padding=0)

            #task_4: joint_type
            self.feat4 = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.ReLU(True)
            )
            self.joint_type_layer = nn.Conv1d(128, 4, kernel_size=1, padding=0)

            #task_5: similar matrix
            self.feat5 = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(True)
            )

            #task_6: confidence
            self.feat6 = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.ReLU(True)
            )
            self.confidence_layer = nn.Conv1d(128, 1, kernel_size=1, padding=0)

        elif self.stage == Stage.Stage2:
            pass
        elif self.stage == Stage.Stage3:
            pass
        else:
            raise NotImplementedError(f'No implementation for the stage {self.stage.value}')

    def forward(self, input):
        batch_size = input.size(dim=0)

        features = self.backbone(input)
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

        return pred

    def losses(self, pred, gt):
        # The returned loss is a value
        if self.stage == Stage.stage1:
            mask = gt['anchor_pts'].float()

            # pred['pred_anchor_pts']: B*N*3K, gt['npcs_per_point']: B*N*3, gt_seg_onehot: B*N*K
            anchor_pts_loss = loss.compute_anchor_pts_loss(
                pred['anchor_pts'],
                gt['anchor_pts'],
                mask=mask
            )

            joint_direction_cat_loss = loss.compute_joint_direction_cat_loss(
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

            joint_type_loss = loss.compute_joint_type_loss(
                pred['joint_type'],
                gt['joint_type'],
                mask=mask
            )

            sim_thresh = 80.0
            gt_simmat = gt['simmat']
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
                threshold = sim_thresh
            )

            epsilon = torch.ones(gt_simmat.size(dim=0), gt_simmat.size(dim=1)).float() * 1e-6
            epsilon = epsilon.to(self.device)
            confidence_loss = loss.compute_confidence_loss(
                pred['confidence'],
                pred['simmat'],
                gt_simmat,
                threshold = sim_thresh,
                epsilon = epsilon
            )

            loss_dict = {
                'anchor_pts_loss': anchor_pts_loss,
                'joint_direction_cat_loss': joint_direction_cat_loss,
                'joint_direction_reg_loss': joint_direction_reg_loss,
                'joint_origin_reg_loss': joint_origin_reg_loss,
                'joint_type_loss': joint_type_loss,
                'simmat_loss': simmat_loss,
                'confidence_loss': confidence_loss,
            }

        return loss_dict
