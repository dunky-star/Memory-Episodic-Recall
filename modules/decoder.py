# modules/decoder.py

import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder head f_θ concatenates latent embedding z_t and recall embedding r_t,
    then outputs class logits.

    Following your Core Math:
      • Inputs: z_t ∈ ℝᵈ, r_t ∈ ℝᵈ
      • Concatenate → h_t ∈ ℝ²ᵈ
      • f_θ(h_t) = W₂·ReLU(W₁·h_t + b₁) + b₂ = logits
    """
    def __init__(self, latent_dim: int, hidden_dim: int, num_classes: int):
        super(Decoder, self).__init__()
        # Linear₁: projects 2d → hidden_dim
        self.fc1 = nn.Linear(latent_dim * 2, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        # Linear₂: projects hidden_dim → num_classes (logits)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
          z (torch.Tensor): [B, d], latent embedding from Encoder
          r (torch.Tensor): [B, d], recall embedding from RecallEngine
        Returns:
          logits (torch.Tensor): [B, num_classes]
        """
        # 1) concatenate z_t and r_t → [B, 2d]
        h = torch.cat([z, r], dim=1)
        # 2) hidden projection + non-linearity
        h = self.act(self.fc1(h))
        # 3) final linear to logits
        logits = self.fc2(h)
        return logits
