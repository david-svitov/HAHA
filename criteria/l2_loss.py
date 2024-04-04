import torch

from main import TrainingStage
from utils.data import get_lossmask


class L2Loss(torch.nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self._weight = weight
        self._loss_fn_l2 = torch.nn.MSELoss()

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
            loss = self._loss_fn_l2(gt_image * loss_mask, render_image * loss_mask) * self._weight
        else:
            loss = self._loss_fn_l2(gt_image, render_image) * self._weight
        return loss
