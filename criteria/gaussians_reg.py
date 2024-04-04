import torch
from pytorch3d.ops import knn_points

from main import TrainingStage
from utils.general import o_act


class OpacityReg(torch.nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self._weight = weight

    def forward(self, data_dict, training_stage):
        if training_stage not in [TrainingStage.OPTIMIZE_OPACITY]:
            return 0
        if "gaussians_opacity" not in data_dict:
            return 0
        opacity = data_dict["gaussians_opacity"]
        #xyz = data_dict["offset_xyz"]
        #y = xyz[:, :, 1]
        #opacity /= torch.abs(y)
        loss = torch.abs(opacity).mean() * self._weight
        return loss


class PoseReg(torch.nn.Module):
    def __init__(self, weight_inside=1., weight_abs=1.):
        super().__init__()
        self._weight_inside = weight_inside
        self._weight_abs = weight_abs

    def forward(self, data_dict, training_stage):
        if training_stage not in [TrainingStage.OPTIMIZE_GAUSSIANS]:
            return 0
        if "offset_xyz" not in data_dict:
            return 0
        xyz = data_dict["offset_xyz"]
        y = xyz[:, :, 1]
        loss = torch.nn.functional.relu(-y).mean() * self._weight_inside
        loss += torch.abs(y).mean() * self._weight_abs
        return loss


class KNNReg(torch.nn.Module):
    def __init__(self, K=3, weight=1.):
        super().__init__()
        self._weight = weight
        self._K = K

    def forward(self, data_dict, training_stage):
        if training_stage not in [TrainingStage.OPTIMIZE_GAUSSIANS, TrainingStage.OPTIMIZE_OPACITY]:
            return 0
        if "gaussians_xyz" not in data_dict:
            return 0
        batch_size = len(data_dict["gaussians_xyz"])
        loss = 0
        for i in range(batch_size):
            xyz = data_dict["gaussians_xyz"][i]
            rotations = data_dict["gaussians_rotations"][i]
            get_s = data_dict["gaussians_scales"][i]
            features_rest = data_dict["gaussians_colors"][i]
            get_o = data_dict["gaussians_opacity"][i]

            dist_sq, nn_ind, _ = knn_points(xyz[None], xyz[None], K=self._K, return_nn=False)
            nn_ind = nn_ind.squeeze(0)
            # reg the std inside knn
            q = rotations[nn_ind, :]  # N,K,4
            s = get_s[nn_ind, :]  # N,K,3
            o = get_o[nn_ind, :]  # N,K,1
            q_std = q.std(dim=1).mean()
            s_std = s.std(dim=1).mean()
            o_std = o.std(dim=1).mean()

            ch = features_rest[nn_ind, :]  # N,K,C
            ch_std = ch.std(dim=1).mean()
            if ch.shape[-1] == 0:
                ch_std = torch.zeros_like(ch_std)

            loss += (q_std + s_std + o_std + ch_std + dist_sq.mean())

            if "warping_xyz" in data_dict:
                xyz = data_dict["warping_xyz"][i]
                rotations = data_dict["warping_rotations"][i]
                get_s = data_dict["warping_scales"][i]
                q = rotations[nn_ind, :]  # N,K,4
                s = get_s[nn_ind, :]  # N,K,3
                p = xyz[nn_ind, :]  # N,K,3
                q_std = q.std(dim=1).mean()
                s_std = s.std(dim=1).mean()
                p_std = p.std(dim=1).mean()

                loss += (q_std + s_std + p_std)

        loss = loss / batch_size  * self._weight
        if training_stage in [TrainingStage.OPTIMIZE_OPACITY]:
            loss /= 10  # Reduce KNN impact on the last stage
        return loss
