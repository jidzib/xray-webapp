import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalCheXpertModel(nn.Module):
    def __init__(self, num_tabular_features, num_classes=14, backbone="densenet121", pretrained=False):
        super().__init__()

        if backbone == "efficientnet_v2_s":
            self.cnn = models.efficientnet_v2_s(pretrained=pretrained)
            in_features = self.cnn.classifier[1].in_features
            self.cnn.classifier = nn.Identity()

        elif backbone == "densenet121":
            self.cnn = models.densenet121(pretrained=pretrained)
            in_features = self.cnn.classifier.in_features
            self.cnn.classifier = nn.Identity()

        elif backbone == "resnet34":
            self.cnn = models.resnet34(pretrained=pretrained)
            in_features = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()

        elif backbone == "resnet50":
            self.cnn = models.resnet50(pretrained=pretrained)
            in_features = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()

        else:
            raise ValueError(backbone)

        self.tabular_net = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, tabular):
        img_features = self.cnn(image)
        tab_features = self.tabular_net(tabular)
        fused = torch.cat([img_features, tab_features], dim=1)
        return self.classifier(fused)
