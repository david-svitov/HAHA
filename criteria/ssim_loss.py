import torch

from main import TrainingStage
from utils.data import get_lossmask
from utils.ssim import ssim


class SSIMLoss(torch.nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self._weight = weight

    def forward(self, data_dict, training_stage):
        if training_stage not in [TrainingStage.OPTIMIZE_OPACITY,
                                  TrainingStage.OPTIMIZE_GAUSSIANS,
                                  TrainingStage.FINETUNE_TEXTURE,
                                  TrainingStage.INIT_TEXTURE,
                                  TrainingStage.FINETUNE_POSE]:
            return 0
        if training_stage in [TrainingStage.FINETUNE_TEXTURE,
                              TrainingStage.INIT_TEXTURE]:
            use_mask = True
        else:
            use_mask = False
        gt_image = data_dict["rgb_image"]
        render_image = data_dict["rasterization"]

        if use_mask:
            loss_mask = get_lossmask(data_dict["merged_mask"][:, 0:1, ...], data_dict["mask_image"])
            loss = (1.0 - ssim(gt_image * loss_mask, render_image * loss_mask)) * self._weight
        else:
            loss = (1.0 - ssim(gt_image, render_image)) * self._weight
        return loss
