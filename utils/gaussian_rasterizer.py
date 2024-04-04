import math

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


class MyGaussianRasterizer(torch.nn.Module):
    def __init__(
            self,
            render_size=640,
    ):
        super().__init__()

        self._render_size = render_size

    def _focal2fov(self, focal, pixels):
        return 2 * math.atan(pixels / (2 * focal))

    def forward(self, data_dict):
        rendered_image_list = []
        depth_list = []
        alpha_list = []
        radii_list = []
        means2D = torch.zeros_like(
            data_dict["gaussians_xyz"],
            dtype=data_dict["gaussians_xyz"].dtype,
            requires_grad=True,
            device=data_dict["gaussians_xyz"].device,
        )
        means2D.retain_grad()
        for frame_id in range(len(data_dict["gaussians_xyz"])):
            xyz = data_dict["gaussians_xyz"][frame_id]
            colors_precomp = data_dict["gaussians_colors"][frame_id]
            opacity = data_dict["gaussians_opacity"][frame_id]
            scales = data_dict["gaussians_scales"][frame_id]
            rotations = data_dict["gaussians_rotations"][frame_id]
            background = data_dict["background"][frame_id]

            viewmatrix = data_dict["camera_transform"][frame_id].clone()
            projection_matrix = data_dict["camera_matrix"][frame_id].clone()

            viewmatrix = viewmatrix.T
            projection_matrix = projection_matrix.T

            means3D = xyz

            full_proj_transform = (viewmatrix.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = viewmatrix.inverse()[3, :3]
            depths_condition = data_dict["mesh_depth"][frame_id][0]
            INF_FAR = 1000
            depths_condition = (1 - data_dict["mask_uv"][frame_id][0]) * INF_FAR + depths_condition
            depths_condition = depths_condition.detach()

            tanfovx = 1 / projection_matrix[0, 0]
            tanfovy = 1 / projection_matrix[1, 1]

            raster_settings = GaussianRasterizationSettings(
                image_height=self._render_size,
                image_width=self._render_size,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=background,
                scale_modifier=1.0,
                viewmatrix=viewmatrix,
                projmatrix=full_proj_transform,
                sh_degree=0,  # ! use pre-compute color!
                campos=camera_center,
                prefiltered=False,
                debug=False,
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            ret = rasterizer(
                means3D=means3D.float(),
                means2D=means2D[frame_id].float(),
                colors_precomp=colors_precomp.float(),
                opacities=opacity.float(),
                depths_condition=depths_condition.float(),
                scales=scales.float(),
                rotations=rotations.float(),
                cov3D_precomp=None,
            )
            rendered_image, radii, depth, alpha = ret
            rendered_image_list.append(rendered_image)
            depth_list.append(depth)
            alpha_list.append(alpha)
            radii_list.append(radii)

        rendered_image = torch.stack(rendered_image_list, dim=0)
        depth = torch.stack(depth_list, dim=0)
        alpha = torch.stack(alpha_list, dim=0)
        radii = torch.stack(radii_list, dim=0)

        return {
            "gaussian_rasterization": rendered_image,
            "gaussian_depth": depth,
            "gaussian_alpha": alpha,
            "radii": radii,
            "viewspace_points": means2D,
            "visibility_filter": radii > 0,
        }
