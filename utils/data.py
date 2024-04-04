import json
from collections import namedtuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []

    gender_pd = []
    gender_gt = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(body_keypoints)

    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt)


def dict2device(data_dict, device):
    """
    Move all tensors in a dict to the device

    :param data_dict: Dict with tensors
    :param device: Target device
    :return: Pointer to the same dict with all tensors moved on the device
    """
    if hasattr(data_dict, 'to'):
        return data_dict.to(device)

    if type(data_dict) == dict:
        for k, v in data_dict.items():
            data_dict[k] = dict2device(v, device)
    return data_dict


def pass_smplx_dict(smplx_params_dict, smplx_model, device):
    """
    Calculate vertices and joints positions by SMPL-X pose and shape vectors

    :param smplx_params_dict: Dict of SMPL-X parameters (shape, pose)
    :param smplx_model: SMPL-X model to inference with parameters
    :param device: Target device for inference
    :return: Dict with resulting vertices and joints positions
    """
    smplx_input = {}

    for k, v in smplx_params_dict.items():
        if k != 'gender':
            if type(v) != torch.Tensor and type(v) != torch.nn.Parameter:
                smplx_input[k] = torch.FloatTensor(v).to(device)
            else:
                smplx_input[k] = v.to(device)

    smplx_output = smplx_model(**smplx_input)
    vertices = smplx_output.vertices
    joints = smplx_output.joints
    smplx_output_dict = dict(vertices=vertices, joints=joints)
    return smplx_output_dict


def get_lossmask(uv_mask, real_segm):
    with torch.no_grad():
        lossmask = uv_mask.detach().float() * real_segm.detach().float()
    return lossmask


def rotate_verts(verts, angle, axis='z'):
    """
    Rotate vertices around selected axis

    :param verts: Vertices to rotate
    :param angle: Angle in radians
    :param K: Camera parameters matrix
    :param axis: Axis around which to rotate
    :return: Rotated copy of vertices
    """
    if axis == "z":
        angles = [angle, 0, 0]
    if axis == "y":
        angles = [0, angle, 0]
    if axis == "x":
        angles = [0, 0, angle]
    rotation_matrix = Rotation.from_euler("zyx", np.array(angles)).as_matrix()
    rotation_matrix = torch.from_numpy(rotation_matrix).type(torch.float32).to(verts.device).unsqueeze(0)

    verts_rot = verts.clone()

    mean_point = torch.mean(verts_rot, dim=1)
    verts_rot -= mean_point

    verts_rot = verts_rot.bmm(rotation_matrix.transpose(1, 2))

    verts_rot += mean_point
    return verts_rot
