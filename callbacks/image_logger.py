import os
from typing import Optional, Any

import cv2
import numpy as np
import torch
import torchvision.utils

from utils.uv_utils import VideoWriter


class ImageLogger:
    def __init__(self, visualize_step=100):
        super().__init__()
        self._visualize_step = visualize_step
        self._video_frames = []
        self._gt_video_frames = []

    def _make_grid(self, images, normalize=False):
        grid = torchvision.utils.make_grid(images, 4)
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.detach().cpu().numpy()
        if normalize:
            max_v = np.max(grid)
            min_v = np.min(grid[grid > 1])
            grid = (grid - min_v) / (max_v - min_v)
        grid = np.clip(np.rint(grid * 255), 0, 255).astype(np.uint8)
        return grid

    def _log_images_val(self, save_folder, batch, global_step):
        keys_to_save = [
            "mesh_rasterization",
            "rgb_image",
            "gaussian_rasterization",
            "merged_mask",
            "gaussian_depth",
            "mesh_depth",
            "rasterization",
            "skin_mask",
            "gaussian_alpha",
            "texture"
        ]

        filename = "s{:06}_b{:03}.png".format(global_step, batch["batch_idx"])

        os.makedirs(save_folder, exist_ok=True)
        for key_name in keys_to_save:
            if key_name not in batch:
                continue
            data = batch[key_name]
            if key_name in ["texture"]:
                data = data[None]
            if key_name in ["gaussian_depth", "mesh_depth"]:
                grid = self._make_grid(data, normalize=True)
            else:
                grid = self._make_grid(data)

            sub_folder = os.path.join(save_folder, key_name)
            os.makedirs(sub_folder, exist_ok=True)
            cv2.imwrite(os.path.join(sub_folder, filename), grid[..., ::-1])

    def _log_images_test(self, save_folder, batch):
        def tensor_to_image(image):
            image = image.detach().cpu().numpy()
            image = np.clip(image, 0, 1)
            image = np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)
            return image

        keys_to_save = [
            "rgb_image",
            "rasterization",
        ]

        os.makedirs(save_folder, exist_ok=True)
        pids = batch['pid']
        for id, pid in enumerate(pids):
            filename = pid + ".png"

            image_list = []
            for key_name in keys_to_save:
                image = batch[key_name][id]
                sub_folder = os.path.join(save_folder, key_name)
                os.makedirs(sub_folder, exist_ok=True)

                image = torch.permute(image, (1, 2, 0))
                image = tensor_to_image(image)
                image_list.append(image)
                cv2.imwrite(os.path.join(sub_folder, filename), image[..., ::-1])

            sub_folder = os.path.join(save_folder, "result_concat")
            os.makedirs(sub_folder, exist_ok=True)

            pred_rgb = torch.permute(batch["rasterization"][id], (1, 2, 0))
            rgb_gt = torch.permute(batch["rgb_image"][id], (1, 2, 0))

            # Save error map
            errmap = (pred_rgb - rgb_gt).square().sum(-1).sqrt().cpu().numpy() / np.sqrt(3)
            errmap = cv2.applyColorMap((errmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            image_list.append(errmap[..., ::-1])
            result = cv2.hconcat(image_list)
            cv2.imwrite(os.path.join(sub_folder, filename), result[..., ::-1])

            # TODO: store frames for gaussians-only and mesh-only
            self._video_frames.append(tensor_to_image(pred_rgb))
            self._gt_video_frames.append(tensor_to_image(rgb_gt))

    def _visualize_images(self, batch):
        rasterization = self._make_grid(batch["rasterization"])
        rgb_image = self._make_grid(batch["rgb_image"])
        visualize = cv2.vconcat([rasterization, rgb_image])
        cv2.imwrite("visalize.png", visualize[..., ::-1])

    def _save_video(self, video_path, frames):
        result = VideoWriter(video_path, 10)

        for frame in frames:
            # frame = cv2.resize(frame, (540, 540))  # TODO: remove. This one is for compatibility with GART
            result.add_frame(frame)

        result.close()

    def on_train_batch_end(
            self,
            runner,
            outputs
    ):
        if (runner.global_step - 1) % self._visualize_step == 0:
            self._visualize_images(outputs)

    def on_validation_batch_end(
            self,
            runner,
            outputs
    ):
        save_folder = os.path.join(runner.logger.log_dir, 'val')
        self._log_images_val(save_folder, outputs, runner.global_step)

    def on_test_batch_end(
            self,
            runner,
            outputs
    ):
        save_folder = os.path.join(runner.logger.log_dir, 'test')
        self._log_images_test(save_folder, outputs)

    def on_test_end(self, runner):
        save_folder = os.path.join(runner.logger.log_dir, 'video')
        os.makedirs(save_folder, exist_ok=True)

        self._save_video(os.path.join(save_folder, 'renders.mp4'), self._video_frames)
        self._save_video(os.path.join(save_folder, 'gt.mp4'), self._gt_video_frames)

        self._video_frames = []
        self._gt_video_frames = []
