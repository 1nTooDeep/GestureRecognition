from movinets import MoViNet

import torch.nn as nn


class MoviNet(nn.Module):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = MoViNet(cfg, num_classes=11, causal=True, pretrained=False)

    def forward(self, x):
        return self.model.forward(x)
