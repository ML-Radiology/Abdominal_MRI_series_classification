import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

try:
    from config import classes
except ImportError:
    from ..config import classes

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=len(classes)):
        super(CustomResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)

        # Replace the last fully connected layer to match the number of classes
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)


class CustomDenseNet(nn.Module):
    def __init__(self, num_classes=len(classes), pretrained=True):
        super(CustomDenseNet, self).__init__()
        model = models.densenet121(pretrained=pretrained)
        self.densenet = model

        # Replace the last classifier layer to match the number of classes
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.densenet(x)



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
