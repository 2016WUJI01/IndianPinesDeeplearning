import torch
import torch.nn as nn


class SEAttention(nn.Module):
    def __init__(self, channel=64 * 17 * 17, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((17, 17))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()
        y = self.avg_pool(x).view(B, C * H * W)
        y = self.fc(y).view(B, C, H, W)
        out = x * y
        return out


class MulitNet(nn.Module):
    def __init__(self, class_num=16):
        super(MulitNet, self).__init__()
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2d_4 = nn.Sequential(
            nn.Conv2d(576, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.seAttention = SEAttention()
        self.fc1 = nn.Linear(18496, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, class_num)
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv3d_1(x)
        out = self.conv3d_2(out)
        out = self.conv3d_3(out)
        out = self.conv2d_4(out.reshape(out.shape[0], -1, 19, 19))
        out = self.seAttention(out)
        out = out.reshape(out.shape[0], -1)
        out = self.relu(self.dropout(self.fc1(out)))
        out = self.relu(self.dropout(self.fc2(out)))
        out = self.fc3(out)
        return out
