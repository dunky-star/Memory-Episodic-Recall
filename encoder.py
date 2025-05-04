import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Transforms input images into a latent embedding z.
    """
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 28×28 → 28×28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28×28 → 14×14
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 14×14 → 14×14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14×14 → 7×7
            nn.Flatten(),  # → (64*7*7)
        )
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        z = self.fc(features)
        return z