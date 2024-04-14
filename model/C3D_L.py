import torch
import torch.nn as nn


class C3D_L(nn.Module):
    def __init__(self, num_classes=11):
        super(C3D_L, self).__init__()
        self.c3d_model = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            #  output: 16 * 56 * 56
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # output: 8 * 28 * 28
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # output: 4 * 14 * 14
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # output: 2 * 7 * 7
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # output: [256 * 3 * 3]
        )
        self.rnn_input_dim = 256 * 1 * 3 * 3

        # RNN 部分
        self.rnn_hidden_units = 512
        self.num_rnn_layers = 1
        self.rnn = nn.GRU(self.rnn_input_dim, self.rnn_hidden_units, self.num_rnn_layers, batch_first=True)

        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(self.rnn_hidden_units, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
            nn.Dropout(p=0.5)
        )

    def segment0(self, x):
        batch_size, channels, sequence_length, _, _ = x.shape
        c3d_features = []
        for i in range(sequence_length):
            feature = x[:, :, i]
            out = self.c3d_model(feature)
            c3d_features.append(out)
        c3d_features = torch.stack(c3d_features, dim=1)
        c3d_features = c3d_features.reshape(c3d_features.shape[0],
                                            c3d_features.shape[1],
                                            -1)
        return c3d_features

    def segment1(self, x):
        x, _ = self.rnn(x)
        # 取出 RNN 最后一层的隐藏状态作为整个视频序列的特征表示
        x = x[:, -1, :]  # 取最后一个时间步的输出
        return x

    def segment2(self, x):
        # class_logits =
        return self.classifier(x)

    def forward(self, x):
        x = self.segment0(x)
        x = self.segment1(x)
        return self.segment2(x)