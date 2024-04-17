import torch
from pprint import pprint
import sys
from timesformer_pytorch import TimeSformer

class Timesformer(torch.nn.Module):
    def __init__(self):
        super(Timesformer,self).__init__()
        self.model = TimeSformer(
        dim=256,
        image_size=112,
        patch_size=16,
        num_frames=30,
        channels=3,
        num_classes=11,
        depth=3,
        heads=8,
        attn_dropout=0.
    )

    def forward(self, x):
        return self.model(x)