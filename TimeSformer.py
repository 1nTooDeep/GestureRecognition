import torch
import torch_directml
from pprint import pprint
import sys
from timesformer_pytorch import TimeSformer

class Timesformer:
    def __init__(self):
        self.model = TimeSformer(
        dim=256,
        image_size=112,
        patch_size=16,
        num_frames=30,
        channels=24,
        num_classes=11,
        depth=1,
        heads=8,
        attn_dropout=0.
    )

    def forward(self, x):
        return self.model(x)