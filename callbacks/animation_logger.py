import math
import os
import statistics
import time

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from utils.uv_utils import VideoWriter


class AnimationLogger:
    def __init__(self, animation_sequence='/mounted//home/dsvitov/Datasets/AMASS/CMU/61/61_15_stageii.npz'):
        self._animation_sequence = animation_sequence

    def _save_video(self, video_path, frames):
        result = VideoWriter(video_path, 60)

        for frame in frames:
            result.add_frame(frame)

        result.close()

    def on_train_batch_end(self, runner, outputs):
        pass

    def on_validation_batch_end(self, runner, outputs):
        pass

    def on_test_batch_end(self, runner, outputs):
        pass

    def on_test_end(self, runner):
        save_folder = os.path.join(runner.logger.log_dir, 'video')
        os.makedirs(save_folder, exist_ok=True)

        video_frames = self._render_animation(runner)
        self._save_video(os.path.join(save_folder, 'animation.mp4'), video_frames)

        save_folder = os.path.join(runner.logger.log_dir, 'video', 'frames')
        os.makedirs(save_folder, exist_ok=True)

        for num, frame in enumerate(video_frames):
            cv2.imwrite(os.path.join(save_folder, str(num) + '.png'), frame[..., ::-1])

    def _render_animation(self, runner):
        animation_data = np.load(self._animation_sequence)
        frames = []
        data_dict = {}
        data_dict["pid"] = ["000"]

        rotation_offset = Rotation.from_euler("zyx", np.array([math.pi, 0, math.pi / 2])).as_matrix()
        rotation_offset = torch.tensor(rotation_offset)

        camera_matrix = torch.tensor([[[4.9270 * 3, 0.0000, -0.0519, 0.0000],
                                       [0.0000, 4.9415 * 3, 0.0000, 0.0000],
                                       [0.0000, 0.0000, 1.0001, -0.0101],
                                       [0.0000, 0.0000, 1.0000, 0.0000]]], device='cuda')
        camera_transform = torch.tensor([[[1., 0., 0., 0.],
                                          [0., 1., 0., 0.9],
                                          [0., 0., 1., 15.2122],
                                          [0., 0., 0., 1.]]], device="cuda")

        camera_transform[:, :3, :3] = rotation_offset
        expression = torch.zeros((1, 1, 10), device="cuda")

        frames_time = []
        n_frames = len(animation_data["trans"])
        for i in tqdm(range(n_frames)):
            smplx_params = {}
            smplx_params["transl"] = torch.FloatTensor([animation_data["trans"][i:i + 1]])
            smplx_params["global_orient"] = torch.FloatTensor([animation_data["root_orient"][i:i + 1]])
            smplx_params["body_pose"] = torch.FloatTensor([animation_data["pose_body"][i:i + 1]])
            smplx_params["right_hand_pose"] = torch.FloatTensor([animation_data["pose_hand"][i:i + 1, :12]])
            smplx_params["left_hand_pose"] = torch.FloatTensor([animation_data["pose_hand"][i:i + 1, 45:45 + 12]])
            smplx_params["leye_pose"] = torch.FloatTensor([animation_data["pose_eye"][i:i + 1, :3]])
            smplx_params["reye_pose"] = torch.FloatTensor([animation_data["pose_eye"][i:i + 1, 3:]])
            smplx_params["jaw_pose"] = torch.FloatTensor([animation_data["pose_jaw"][i:i + 1]])
            smplx_params["expression"] = expression
            smplx_params["betas"] = [runner._betas]
            smplx_params["camera_matrix"] = camera_matrix
            smplx_params["camera_transform"] = camera_transform

            smplx_params["transl"] = smplx_params["transl"].to(runner.device)
            smplx_params["global_orient"] = smplx_params["global_orient"].to(runner.device)
            smplx_params["body_pose"] = smplx_params["body_pose"].to(runner.device)
            smplx_params["right_hand_pose"] = smplx_params["right_hand_pose"].to(runner.device)
            smplx_params["left_hand_pose"] = smplx_params["left_hand_pose"].to(runner.device)

            data_dict["smplx_params"] = smplx_params
            data_dict["background"] = torch.ones([3, runner._render_size, runner._render_size], device="cuda")
            with torch.no_grad():
                start = time.time()
                runner._render_frame(data_dict, runner._loaded_training_stage, optimize_pose=False)
                stop = time.time()
                if i > 10:
                    frames_time.append(stop - start)

            frame = data_dict["rasterization"][0]

            frame = frame.detach().cpu().numpy()
            frame = np.transpose(frame, (1, 2, 0))
            frame = np.clip(frame, 0, 1) * 255
            frame = frame.astype(np.uint8).copy()

            frames.append(frame)
        avg_time = statistics.mean(frames_time)
        print('FPS:', 1 / avg_time)
        return frames
