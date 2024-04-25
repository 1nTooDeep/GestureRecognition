import torch
import torch.nn as nn


class C3D(nn.Module):
    def __init__(self, num_classes=11):
        super(C3D, self).__init__()
        self.feature = nn.Sequential(
            nn.BatchNorm3d(3),
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 1, 1)),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Flatten(start_dim=1, end_dim=-1)
        )
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.Dropout(p=0.5)
        )


    def forward(self, x):
        x = self.feature(x)
        return self.classifier(x)
    
    
    
if __name__ == '__main__':
    model = C3D()
    for i in range(100):
        x = torch.rand(8, 3, 30, 112, 112)
        print(model(x))
