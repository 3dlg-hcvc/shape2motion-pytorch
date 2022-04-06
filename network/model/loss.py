import torch.nn.functional as F
import torch

def compute_anchor_pts_loss(pred_anchor_pts, gt_anchor_pts, mask):
    negative_mask = torch.ones_like(mask) - mask
    num_pos = torch.unsqueeze(torch.sum(mask, 1), 1)
    num_neg = torch.unsqueeze(torch.sum(negative_mask, 1), 1)
    num_points = pred_anchor_pts.size(dim=2)

    anchor_pts_loss = torch.mean(F.cross_entropy(pred_anchor_pts, gt_anchor_pts.long(), reduction='none') * (mask * (num_neg/num_pos) + 1))
    true_anchor_pos = torch.sum(torch.eq(torch.argmax(pred_anchor_pts, axis=1).int(), gt_anchor_pts.int()).float() * mask, axis = 1)
    true_pts_pos = torch.sum(torch.eq(torch.argmax(pred_anchor_pts, axis=1).int(), gt_anchor_pts.int()).float(), axis = 1)
    anchor_pts_recall = torch.mean(true_anchor_pos / torch.sum(mask,axis=1))
    anchor_pts_accuracy = torch.mean(true_pts_pos / num_points)
    return anchor_pts_loss, anchor_pts_recall, anchor_pts_accuracy

def compute_joint_direction_cat_loss(pred_joint_direction_cat, gt_joint_direction_cat, mask):
    mask_sum = torch.sum(mask, axis=1)
    joint_direction_cat_loss = torch.mean(torch.sum(F.cross_entropy(pred_joint_direction_cat, gt_joint_direction_cat.long(), reduction='none') * mask, axis=1) / mask_sum)
    joint_direction_cat_accuracy = torch.mean(torch.sum(torch.eq(torch.argmax(pred_joint_direction_cat, axis=1).int(), gt_joint_direction_cat.int()).float()*mask, axis=1) / mask_sum)
    return joint_direction_cat_loss, joint_direction_cat_accuracy

def compute_joint_direction_reg_loss(pred_joint_direction_reg, gt_joint_direction_reg, mask):
    joint_direction_reg_loss = torch.mean(torch.sum(torch.mean(F.smooth_l1_loss(pred_joint_direction_reg, gt_joint_direction_reg, reduction='none'), axis=2) * mask, axis=1) / torch.sum(mask, axis=1))
    return joint_direction_reg_loss
    

def compute_joint_origin_reg_loss(pred_joint_origin_reg, gt_joint_origin_reg, mask):
    joint_origin_reg_loss = torch.mean(torch.sum(torch.mean(F.smooth_l1_loss(pred_joint_origin_reg, gt_joint_origin_reg, reduction='none'), axis=2) * mask, axis=1) / torch.sum(mask, axis=1))
    return joint_origin_reg_loss

def compute_joint_type_loss(pred_joint_type, gt_joint_type, mask):
    joint_type_loss = torch.mean(torch.sum(F.cross_entropy(pred_joint_type, gt_joint_type.long(), reduction='none') * mask, axis=1) / torch.sum(mask, axis=1))
    joint_type_accuracy = torch.mean(torch.sum(torch.eq(torch.argmax(pred_joint_type, axis=1).int(), gt_joint_type.int()).float() * mask,axis = 1) / torch.sum(mask,axis=1))
    return joint_type_loss, joint_type_accuracy

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

def compute_motion_scores_loss(pred_motion_scores, gt_motion_scores, mask, epsilon):
    motion_scores_loss = torch.mean(torch.sum(torch.mean(F.smooth_l1_loss(pred_motion_scores, gt_motion_scores, reduction='none'), axis=2) * mask, axis=1) / (torch.sum(mask, axis=1) + epsilon))
    return motion_scores_loss

def compute_part_proposal_loss(pred_part_proposal, gt_part_proposal, epsilon):
    num_points = pred_part_proposal.size(dim=2)
    part_proposal_loss = F.cross_entropy(pred_part_proposal, gt_part_proposal.long(), reduction='mean')
    proposal = torch.argmax(pred_part_proposal, axis=1).int()
    part_proposal_accuracy = torch.mean(torch.sum(torch.eq(proposal, gt_part_proposal.int()).float(), axis = 1) / num_points)
    
    proposal = torch.gt(proposal.float(), 0.5)
    gt_proposal = torch.gt(gt_part_proposal.float(), 0.5)
    iou = torch.mean(torch.sum(torch.logical_and(proposal, gt_proposal).float(), axis=-1) / (torch.sum(torch.logical_or(proposal, gt_proposal).float(), axis=-1) + epsilon))
    return part_proposal_loss, part_proposal_accuracy, iou

def compute_motion_regression_loss(pred_motion_regression, gt_motion_regression):
    motion_regression_loss = torch.mean(torch.sum(F.smooth_l1_loss(pred_motion_regression, gt_motion_regression, reduction='none'), axis=1))
    return motion_regression_loss






