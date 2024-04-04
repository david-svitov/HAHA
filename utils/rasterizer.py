import nvdiffrast.torch as dr
import torch


class NVDiffrast(torch.nn.Module):
    def __init__(self, render_size=640):
        super().__init__()
        self._cudactx = dr.RasterizeCudaContext()
        self._render_size = render_size
        self._faces = None
        self._uv_idx = None
        self._uv = None

    def set_smplx_parameters(self, faces, uv, uv_idx):
        self._faces = faces
        self._uv = uv
        self._uv_idx = uv_idx

    def _precalc_points(self, points, camera_mat, camera_transform):
        device = points.device

        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)

        projected_points = torch.einsum('bki,bji->bjk', [camera_transform, points_h])
        projected_points = torch.einsum('bki,bji->bjk', [camera_mat, projected_points])
        return projected_points

    def forward(self, data_dict):
        vertices = data_dict["vertices"]
        camera_mat = data_dict["camera_matrix"]
        camera_transform = data_dict["camera_transform"]
        texture = torch.permute(data_dict["texture"], (1, 2, 0)).contiguous()
        pos = self._precalc_points(vertices, camera_mat, camera_transform).contiguous()
        white_color = torch.ones_like(vertices)
        pos_color = pos
        tri = self._faces.to(pos.device)

        rast, _ = dr.rasterize(self._cudactx, pos, tri, resolution=[self._render_size, self._render_size])

        batch_size = len(vertices)
        uv = self._uv[None, ...].repeat(batch_size, 1, 1).to(rast.device)
        uv_idx = self._uv_idx.to(rast.device)
        texc, _ = dr.interpolate(uv, rast, uv_idx)
        color = dr.texture(texture[None, ...], texc, filter_mode='linear')

        depth, _ = dr.interpolate(pos_color, rast, tri)
        depth = depth[..., 2:3]+0.05  # Z coordinate TODO: understand why we have to use this epsilon

        mask, _ = dr.interpolate(white_color, rast, tri)
        mask = dr.antialias(mask, rast, pos, tri)
        color *= mask

        return {
            "mesh_rasterization": torch.permute(color, (0, 3, 1, 2)),
            "mesh_depth": torch.permute(depth, (0, 3, 1, 2)),
            "mask_uv": torch.permute(mask, (0, 3, 1, 2)),
            "projected_vertices": pos,
        }
