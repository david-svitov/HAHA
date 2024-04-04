import torch

from main import TrainingStage
from utils.data import get_lossmask


def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


class DiceLoss(torch.nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self._weight = weight
        self._loss_fn_l2 = torch.nn.MSELoss()

    def forward(self, data_dict, training_stage):
        if training_stage not in [
            TrainingStage.OPTIMIZE_GAUSSIANS,
            TrainingStage.OPTIMIZE_OPACITY,
            TrainingStage.FINETUNE_POSE
        ]:
            return 0
        gt_mask = data_dict["mask_image"]
        merged_mask = data_dict["merged_mask"][:, 0:1, ...]
        if training_stage in [TrainingStage.OPTIMIZE_GAUSSIANS]:
            loss = dice_loss(1 - merged_mask, 1 - gt_mask) * self._weight
        else:
            loss = dice_loss(merged_mask, gt_mask) * self._weight
        return loss
