import gorilla
import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from typing import Optional
from torch.nn.utils.rnn import pad_sequence
from typing import Union


def get_iou(inputs: torch.Tensor, targets: torch.Tensor, pad_mask: Union[torch.Tensor, None]=None, pred_confidence=0.5):
    '''
    padding modified
    '''
    if pad_mask is not None:
        inputs = inputs.sigmoid()*pad_mask
    else:
        inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= pred_confidence)
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score

def get_iou_prob(inputs: torch.Tensor, targets: torch.Tensor, pad_mask: Union[torch.Tensor, None]=None):
    '''
    padding modified
    '''
    if pad_mask is not None:
        inputs = inputs*pad_mask
    else:
        inputs = inputs
    # thresholding
    binarized_inputs = (inputs >= 0.5)#.float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score

@torch.jit.script
def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()


#@torch.jit.script
def dice_loss(
    inputs,
    targets,
    pad_mask
):
    """
    padding modified
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pad_mask: A float tensor with the same shape as inputs. Stores the binary, 0 for padding, 1 for non-padding.
    """
    if pad_mask is not None:
        inputs = inputs.sigmoid()*pad_mask
    else:
        inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)  # why+1？
    return loss.mean()

@torch.jit.script
def dice_loss_prob(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pad_mask: Union[torch.Tensor, None]=None
):
    """
    padding modified
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pad_mask: A float tensor with the same shape as inputs. Stores the binary, 0 for padding, 1 for non-padding.
    """
    if pad_mask is not None:
        inputs = inputs*pad_mask
    else:
        inputs = inputs
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()

class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #proposals, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #proposals) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #proposals, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        weights = weights.unsqueeze(-1)
        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


@gorilla.LOSSES.register_module()
class Criterion(nn.Module):

    def __init__(
        self,
        loss_weight=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_fun='bce'
    ):
        super().__init__()
        self.loss_fun = loss_fun
        loss_weight = torch.tensor(loss_weight)
        self.register_buffer('loss_weight', loss_weight)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def compute_bce_loss(self, pred_mask, target_mask, valid_masks):
        mask_bce_loss = F.binary_cross_entropy_with_logits(pred_mask, target_mask.float(), reduction='none')
        mask_bce_loss = (mask_bce_loss*valid_masks).sum(-1) / valid_masks.sum(-1)
        mask_bce_loss = mask_bce_loss.mean()
        
        return mask_bce_loss
    
    def forward(self, pred, gt_spmasks, gt_anchor_spmasks, gt_rotation_angles):
        loss_out = {}
        
        # score loss, NOTE: score loss is not applicable in our task, since we are predicting specific masks (placement and anchors)
        # we do not have proposals to score/rank them
        score_loss = 0.0
        
        ################################################
        # Placement Seg Losses
        ################################################
        placement_pred_masks = pred['placement_masks'].squeeze()
        valid_masks = ~pred['batch_mask'].squeeze()
        tgt_padding = pad_sequence(gt_spmasks, batch_first=True)
        
        mask_bce_loss = self.compute_bce_loss(placement_pred_masks, tgt_padding, valid_masks)
        mask_dice_loss = dice_loss(placement_pred_masks, tgt_padding.float(), valid_masks)
        loss_out['mask_bce_loss'] = mask_bce_loss
        loss_out['mask_dice_loss'] = mask_dice_loss
        
        if 'aux_placement_masks' in pred:
            for layer in range(len(pred['aux_placement_masks'])):
                aux_placement_masks = pred['aux_placement_masks'][layer].squeeze()
                
                aux_mask_bce_loss = self.compute_bce_loss(aux_placement_masks, tgt_padding, valid_masks)
                aux_mask_dice_loss = dice_loss(aux_placement_masks, tgt_padding.float(), valid_masks)
                
                mask_bce_loss += aux_mask_bce_loss
                mask_dice_loss += aux_mask_dice_loss
                
                loss_out[f"{layer}_mask_bce_loss"] = aux_mask_bce_loss
                loss_out[f"{layer}_mask_dice_loss"] = aux_mask_dice_loss
        
        ################################################        
        # Rotation BCE loss
        ################################################
        rotation_loss = 0.0
        loss_out['rotation_loss'] = -1
        if 'rotation_angles_logits' in pred:
            rot_tgt_padding = pad_sequence(gt_rotation_angles, batch_first=True)
            rotation_loss = F.binary_cross_entropy_with_logits(pred['rotation_angles_logits'], rot_tgt_padding, reduction='none')
            rot_valid_masks = valid_masks.unsqueeze(-1).repeat(1, 1, 8)
            rotation_loss = (rotation_loss*rot_valid_masks).sum(-1) / rot_valid_masks.sum(-1)
            rotation_loss = rotation_loss.mean()
            loss_out['rotation_loss'] = rotation_loss
                    
        ################################################
        # Anchor Losses
        ################################################
        anchor_mask_dice_loss = 0.0
        anchor_mask_bce_loss = 0.0
        loss_out['anchor_mask_bce_loss'] = -1
        loss_out['anchor_mask_dice_loss'] = -1
        if 'anchor_masks' in pred:
            anchor_pred_masks = pred['anchor_masks'].squeeze()
            anchor_tgt_padding = pad_sequence(gt_anchor_spmasks, batch_first=True)
            
            anchor_mask_bce_loss = self.compute_bce_loss(anchor_pred_masks, anchor_tgt_padding, valid_masks)
            anchor_mask_dice_loss = dice_loss(anchor_pred_masks, anchor_tgt_padding.float(), valid_masks)
            
            if 'aux_anchor_masks' in pred:
                for layer in range(len(pred['aux_anchor_masks'])):
                    aux_anchor_masks = pred['aux_anchor_masks'][layer].squeeze()
                    
                    aux_anchor_mask_bce_loss = self.compute_bce_loss(aux_anchor_masks, anchor_tgt_padding, valid_masks)
                    aux_anchor_mask_dice_loss = dice_loss(aux_anchor_masks, anchor_tgt_padding.float(), valid_masks)
                    
                    anchor_mask_bce_loss += aux_anchor_mask_bce_loss
                    anchor_mask_dice_loss += aux_anchor_mask_dice_loss
                    
                    loss_out[f"{layer}_anchor_mask_bce_loss"] = aux_anchor_mask_bce_loss
                    loss_out[f"{layer}_anchor_mask_dice_loss"] = aux_anchor_mask_dice_loss
            
            loss_out['anchor_mask_bce_loss'] = anchor_mask_bce_loss
            loss_out['anchor_mask_dice_loss'] = anchor_mask_dice_loss
        
        loss = 0.0
        loss += rotation_loss        
        loss += (self.loss_weight[0] * mask_bce_loss + self.loss_weight[1] * mask_dice_loss)
        loss += (self.loss_weight[0] * anchor_mask_bce_loss + self.loss_weight[1] * anchor_mask_dice_loss)
        loss_out['loss'] = loss

        return loss, loss_out
