import torch

from main import TrainingStage
from torchmetrics.image import TotalVariation


class TVLoss(torch.nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self._weight = weight
        self._tv_loss = TotalVariation()

    def forward(self, data_dict, training_stage):
        if training_stage not in [TrainingStage.FINETUNE_TEXTURE,
                                  TrainingStage.INIT_TEXTURE]:
            return 0
        texture = torch.unsqueeze(data_dict["texture"], 0)
        self._tv_loss.to(texture.device)
        loss = self._tv_loss(texture) / torch.numel(texture) * self._weight
        return loss
