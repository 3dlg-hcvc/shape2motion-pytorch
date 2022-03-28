import torch.nn.functional as F
import torch
import pdb

def compute_anchor_pts_loss(pred_anchor_pts, gt_anchor_pts, mask):
    negative_mask = torch.ones_like(mask) - mask
    num_pos = torch.unsqueeze(torch.sum(mask, 1), 1)
    num_neg = torch.unsqueeze(torch.sum(negative_mask, 1), 1)
    num_points = pred_anchor_pts.size(dim=1)

    anchor_pts_loss = torch.mean(F.cross_entropy(pred_anchor_pts, gt_anchor_pts.long(), reduction='none') * (mask * (num_pos/num_neg) + 1))
    true_pos = torch.sum(torch.eq(torch.argmax(pred_anchor_pts, axis=1).int(), gt_anchor_pts.int()).float() * mask, axis = 1)
    anchor_pts_recall = torch.mean(true_pos / torch.sum(mask,axis=1))
    anchor_pts_accuracy = torch.mean(true_pos / num_points)
    return anchor_pts_loss, anchor_pts_recall, anchor_pts_accuracy

def compute_joint_direction_cat_loss(pred_joint_direction_cat, gt_joint_direction_cat, mask):
    mask_sum = torch.sum(mask, axis=1)
    joint_direction_cat_loss = torch.mean(torch.sum(F.cross_entropy(pred_joint_direction_cat, gt_joint_direction_cat.long(), reduction='none') * mask, axis=1) / mask_sum)
    joint_direction_cat_acc = torch.mean(torch.sum(torch.eq(torch.argmax(pred_joint_direction_cat, axis=1).int(), gt_joint_direction_cat.int()).float()*mask, axis=1) / mask_sum)
    return joint_direction_cat_loss, joint_direction_cat_acc

def compute_joint_direction_reg_loss(pred_joint_direction_reg, gt_joint_direction_reg, mask):
    joint_direction_reg_loss = torch.mean(torch.sum(torch.mean(F.smooth_l1_loss(pred_joint_direction_reg, gt_joint_direction_reg, reduction='none'), axis=2) * mask, axis=1) / torch.sum(mask, axis=1))
    return joint_direction_reg_loss
    

def compute_joint_origin_reg_loss(pred_joint_origin_reg, gt_joint_origin_reg, mask):
    joint_origin_reg_loss = torch.mean(torch.sum(torch.mean(F.smooth_l1_loss(pred_joint_origin_reg, gt_joint_origin_reg, reduction='none'), axis=2) * mask, axis=1) / torch.sum(mask, axis=1))
    return joint_origin_reg_loss

def compute_joint_type_loss(pred_joint_type, gt_joint_type, mask):
    joint_type_loss = torch.mean(torch.sum(F.cross_entropy(pred_joint_type, gt_joint_type.long(), reduction='none') * mask, axis=1) / torch.sum(mask, axis=1))
    joint_type_acc = torch.mean(torch.sum(torch.eq(torch.argmax(pred_joint_type, axis=1).int(), gt_joint_type.int()).float() * mask,axis = 1) / torch.sum(mask,axis=1))
    return joint_type_loss, joint_type_acc

def compute_simmat_loss(pred_simmat, gt_simmat, neg_simmat, threshold):
    pos = pred_simmat * (gt_simmat + gt_simmat.transpose(1, 2))
    neg = torch.clamp(threshold - pred_simmat, min=0) * neg_simmat
    simmat_loss = torch.mean(pos + neg)
    return simmat_loss

def compute_confidence_loss(pred_confidence, pred_simmat, gt_simmat, threshold, epsilon):
    gt_positive = torch.gt(gt_simmat, 0.5)
    pred_positive = torch.lt(pred_simmat, threshold)

    pts_iou = torch.sum(torch.logical_and(pred_positive, gt_positive).float(), axis=2) / (torch.sum(torch.logical_or(pred_positive, gt_positive).float(), axis=2) + epsilon)
    confidence_loss = F.mse_loss(pts_iou, torch.squeeze(pred_confidence, 2))
    return confidence_loss





