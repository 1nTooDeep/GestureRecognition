import torch
from pprint import pprint
import sys
from timesformer_pytorch import TimeSformer
from  transformers.models.timesformer import TimesformerConfig,TimesformerModel,TimesformerForVideoClassification


class Timesformer(torch.nn.Module):
    def __init__(self):
        super(Timesformer,self).__init__()
        self.config = TimesformerConfig(
            image_size=112,
            patch_size=16,
            num_channels=3,
            num_frames=30,
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=2,
            hidden_dropout_prob=0.2,
            intermediate_size=512,
            attention_probs_dropout_prob=0.2,
            num_labels = 15,
            problem_type = 'single_label_classification',

        )
        self.model = TimesformerForVideoClassification(self.config)
    

    def forward(self, x, y = None):

        if y is None:
            x = self.model(x)
            x = x.logits
            return x
        else:
            return self.model(x,y)