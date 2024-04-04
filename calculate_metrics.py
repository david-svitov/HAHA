import argparse
from glob import glob

import cv2
import torch
from torch import nn
from torch.cuda.amp import custom_fwd
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class Evaluator(nn.Module):
    """adapted from https://github.com/JanaldoChen/Anim-NeRF/blob/main/models/evaluator.py"""

    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

    # custom_fwd: turn off mixed precision to avoid numerical instability during evaluation
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, rgb, rgb_gt):
        # torchmetrics assumes NCHW format
        rgb = rgb.permute(0, 3, 1, 2).clamp(max=1.0)
        rgb_gt = rgb_gt.permute(0, 3, 1, 2)

        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }


def load_images(folder, device="cuda"):
    imgs = [cv2.imread(fn) for fn in glob(f"{folder}/*.png")]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    imgs = [torch.tensor(img).float().to(device) / 255.0 for img in imgs]
    imgs = torch.stack(imgs)
    return imgs


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-p",
        "--predict",
        type=str,
        const=True,
        default=None,
        nargs="?",
        help="Folder with predicted images",
    )
    parser.add_argument(
        "-g",
        "--ground_truth",
        type=str,
        const=True,
        default=None,
        nargs="?",
        help="Folder with ground truth images",
    )
    return parser


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    device = "cuda"

    imgs_gt = load_images(args.ground_truth, device)
    imgs = load_images(args.predict, device)

    evaluator = Evaluator()
    evaluator = evaluator.to(device)
    evaluator.eval()

    result = evaluator(imgs, imgs_gt)

    print("psnr:", result["psnr"])
    print("ssim:", result["ssim"])
    print("lpips:", result["lpips"])


if __name__ == '__main__':
    main()
