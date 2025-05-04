import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Takes the latent embedding and recall vector, outputs class logits.
    """
    def __init__(self, latent_dim: int = 128, recall_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim + recall_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, z: torch.Tensor, recall_vec: torch.Tensor) -> torch.Tensor:
        x_cat = torch.cat([z, recall_vec], dim=1)
        return self.classifier(x_cat)