# models/custom_cnn.py

import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    """
    Lightweight CNN for federated learning on small datasets.
    Used in many FL papers (e.g., FedAvg, FedProx).
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super(CustomCNN, self).__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier head
        # Assume input size is at least 28x28 → after 2 pools: 7x7
        # For 32x32 → 8x8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8 if in_channels == 3 else 64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        
        self.in_channels = in_channels
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x