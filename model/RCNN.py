import torch
import torch.nn as nn


class RCNN(nn.Module):
    def __init__(self, num_classes):
        super(RCNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(start_dim=1),
            nn.GRU(input_size=25088, hidden_size=256, num_layers=2, batch_first=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        batch_size, frames, channel, W, H = x.size()
        x = x.view(batch_size * frames, channel, W, H)
        output = self.model(x)
        # print(output[0].shape, output[1].shape)
        output = output[0].view(batch_size, frames, -1)
        output = self.classifier(output[:, -1, :])
        return output