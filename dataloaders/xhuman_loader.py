import math
import os
import pickle
from glob import glob

import cv2
import numpy as np
import torch


def path_to_id(path):
    filename = path.split('/')[-1]
    no_extension = filename.split('.')[0]
    id = no_extension.split("_")[-1]
    return id


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def K_to_camera_matrix(K, src_W, src_H):
    scale_y = src_H / src_W
    FoVx = focal2fov(K[0, 0], src_W)
    FoVy = focal2fov(K[1, 1], src_H) / scale_y
    camera_matrix = getProjectionMatrix(znear=0.01, zfar=1.0, fovX=FoVx, fovY=FoVy)
    camera_matrix[2, 2] = 1.0001
    return camera_matrix


class DataLoader(torch.utils.data.Dataset):
    def __init__(
            self,
            data_root,
            use_hashing=False,
            random_background=False,
            white_background=False,
            render_size=640,
    ):
        self._images_path = os.path.join(data_root, "render/image")
        self._masks_path = os.path.join(data_root, "render/depth")
        self._smplx_path = os.path.join(data_root, "SMPLX")
        self._render_size = render_size

        # Get images list
        self._images_list = glob(os.path.join(self._images_path, "*"))
        self._images_list = sorted(self._images_list, key=lambda x: int(path_to_id(x)))
        self._len = len(self._images_list)

        self._random_background = random_background
        self._white_background = white_background
        self._use_hashing = use_hashing
        self._data_hash = {}

        self.sequence_name = data_root.split('/')[-3]

        self._camera_params = dict(np.load(os.path.join(data_root, "render/cameras.npz")))
        self._pid_camera_map = {}
        for num, path in enumerate(self._images_list):
            pid = path_to_id(path)
            self._pid_camera_map[int(pid)] = num

    def load_sample(self, pid):
        rgb_image = cv2.imread(os.path.join(self._images_path, "color_{:06d}.png".format(pid)))
        H, W, _ = rgb_image.shape
        vertical_offset = (H - W) // 2
        rgb_image = rgb_image[vertical_offset:vertical_offset + W]
        rgb_image = cv2.resize(rgb_image, (self._render_size, self._render_size))
        rgb_image = rgb_image[..., ::-1]
        rgb_image = rgb_image.astype(np.float32) / 255.0
        rgb_image = np.transpose(rgb_image, (2, 0, 1))

        if self._white_background:
            background = np.ones_like(rgb_image)
        else:
            background = np.zeros_like(rgb_image)

        if self._random_background:
            background = np.ones_like(rgb_image)
            background_color = np.random.rand(3).astype(np.float32)
            background[0] *= background_color[0]
            background[1] *= background_color[1]
            background[2] *= background_color[2]

        mask_image = cv2.imread(os.path.join(self._masks_path, "depth_{:06d}.tiff".format(pid)), -1)
        mask_image = mask_image[vertical_offset:vertical_offset + W]
        mask_image = cv2.resize(mask_image, (self._render_size, self._render_size))
        mask_image = mask_image.max() - mask_image.astype(np.float32)
        _, mask_image = cv2.threshold(mask_image, 700, 1, cv2.THRESH_BINARY)
        mask_image = mask_image[None]
        rgb_image = mask_image * rgb_image + (1 - mask_image) * background

        # Load pickle
        smplx_filename = os.path.join(self._smplx_path, "mesh-f{:05d}_smplx.pkl".format(pid))
        with open(smplx_filename, "rb") as f:
            smplx_params = pickle.load(f)
        for k in smplx_params.keys():
            smplx_params[k] = smplx_params[k][None]

        smplx_params["camera_matrix"] = K_to_camera_matrix(self._camera_params["intrinsic"], W, H)
        smplx_params["camera_transform"] = self._camera_params["extrinsic"][self._pid_camera_map[pid]]

        smplx_params["camera_matrix"] = smplx_params["camera_matrix"].astype(np.float32)
        smplx_params["camera_transform"] = smplx_params["camera_transform"].astype(np.float32)

        return {
            "pid": str(pid),
            "rgb_image": rgb_image,
            "mask_image": mask_image,
            "background": background,
            "smplx_params": smplx_params,
        }

    def __getitem__(self, index):
        index = index % self._len
        rgb_filename = self._images_list[index]
        pid = int(path_to_id(rgb_filename))

        if rgb_filename in self._data_hash:
            data_dict = self._data_hash[rgb_filename]
        else:
            data_dict = self.load_sample(pid)
            if self._use_hashing:
                self._data_hash[rgb_filename] = data_dict
        return data_dict

    def __len__(self):
        return self._len
