import torch
from pprint import pprint
import sys
from timesformer_pytorch import TimeSformer
from transformers.models.timesformer import TimesformerConfig, TimesformerForVideoClassification


class Timesformer(torch.nn.Module):
    def __init__(self, image_size=112, num_frames=30, intermediate_size=256, num_hidden_layers=4, hidden_size=256):
        super(Timesformer, self).__init__()
        self.config = TimesformerConfig(
            image_size=image_size,
            patch_size=8,
            num_channels=3,
            num_frames=num_frames,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=2,
            hidden_dropout_prob=0.4,
            intermediate_size=intermediate_size,
            attention_probs_dropout_prob=0.4,
            num_labels=14,
            problem_type='single_label_classification',
        )
        self.model = TimesformerForVideoClassification(self.config)

    def forward(self, x, y=None):
        if y is None:
            x = self.model(x)
            x = x.logits
            return x
        else:
            return self.model(x, y)
