import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MosquitoDetector(nn.Module):
    def __init__(self, num_classes=2, in_channels=1, n_mels=64):
        super(MosquitoDetector, self).__init__()

        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 2)),
            ConvBlock(32, 64, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 2)),
            ConvBlock(64, 128, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 2)),
            ConvBlock(128, 256, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 2)),
            ConvBlock(256, 512, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 2)),
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, n_mels, 100)
            output = self.conv_blocks(dummy_input)
            self.feature_dim = output.size(1) * output.size(2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class EfficientMosquitoDetector(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super(EfficientMosquitoDetector, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model(model_name="efficient", num_classes=2):
    if model_name == "efficient":
        return EfficientMosquitoDetector(num_classes=num_classes)
    elif model_name == "simple":
        return MosquitoDetector(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")