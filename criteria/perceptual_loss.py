import random

import cv2
import torch
from lpips import lpips

from main import TrainingStage
from utils.data import get_lossmask


class PerceptualLoss(torch.nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self._weight = weight
        self._loss_fn = lpips.LPIPS(net="vgg")

    def forward(self, data_dict, training_stage):
        if training_stage not in [TrainingStage.OPTIMIZE_OPACITY,
                                  TrainingStage.OPTIMIZE_GAUSSIANS,
                                  TrainingStage.FINETUNE_TEXTURE,
                                  TrainingStage.INIT_TEXTURE]:
            return 0
        if training_stage in [TrainingStage.FINETUNE_TEXTURE,
                              TrainingStage.INIT_TEXTURE]:
            use_mask = True
        else:
            use_mask = False
        self._loss_fn.to(data_dict["rasterization"].device)
        synth_images = data_dict["rasterization"]
        target_images = data_dict["rgb_image"]
        lossmask = get_lossmask(data_dict["merged_mask"][:, 0:1, ...], data_dict["mask_image"])

        sub_synth = []
        sub_target = []
        sum_mask = []
        for image_id in range(len(data_dict["projected_vertices"])):
            verts = data_dict["projected_vertices"][image_id]
            verts_projected = (verts / (verts[:, 3:]))[:, :2]
            point_on_model = random.choice(verts_projected)
            cx, cy = point_on_model
            h, w = synth_images.shape[2:]
            cx = cx * w / 2 + w / 2
            cy = cy * h / 2 + h / 2
            j = max(int(cx), 128)
            i = max(int(cy), 128)

            i = min(i, synth_images.shape[2] - 129)
            j = min(j, synth_images.shape[3] - 129)

            x_slice = slice(j - 128, j + 128)
            y_slice = slice(i - 128, i + 128)
            sub_synth.append(synth_images[image_id, :, y_slice, x_slice])
            sub_target.append(target_images[image_id, :, y_slice, x_slice])
            sum_mask.append(lossmask[image_id, :, y_slice, x_slice])
        """
        vis = target_images[image_id, :, y_slice, x_slice]
        vis = torch.permute(vis, (1, 2, 0))
        vis = vis.detach().cpu().numpy()
        cv2.imshow('debug', vis)
        cv2.waitKey(1)
        """

        sub_synth = torch.stack(sub_synth, dim=0)
        sub_target = torch.stack(sub_target, dim=0)
        sum_mask = torch.stack(sum_mask, dim=0)

        if use_mask:
            loss = self._loss_fn(sub_target * sum_mask, sub_synth * sum_mask).mean() * self._weight
        else:
            loss = self._loss_fn(sub_target, sub_synth).mean() * self._weight
        return loss
