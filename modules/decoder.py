# modules/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    Mixture-of-Experts Decoder for EpiNet.

    Given latent embedding z_t and recall embedding r_t,
    concatenates to h_t ∈ ℝ^{2d}, then uses E parallel experts
    and a gating network to produce class logits:
      • Gate: g = softmax(G·h_t) ∈ ℝ^E
      • Experts: ℓ^{(e)} = expert_e(h_t) ∈ ℝ^K
      • logits = Σ_{e=1}^E g_e · ℓ^{(e)}
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_experts: int = 4
    ):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        # Gating network: 2d → E
        self.gate = nn.Linear(latent_dim * 2, num_experts)
        # Experts: each maps 2d → hidden_dim → num_classes
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_classes)
            ) for _ in range(num_experts)
        ])

    def forward(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
          z (torch.Tensor): [B, d], latent embedding from Encoder
          r (torch.Tensor): [B, d], recall embedding from RecallEngine
        Returns:
          logits (torch.Tensor): [B, num_classes]
        """
        # Concatenate embeddings: [B, 2d]
        h = torch.cat([z, r], dim=1)
        # Compute gating weights: [B, E]
        gate_logits = self.gate(h)
        gate_weights = F.softmax(gate_logits, dim=1)
        # Expert outputs: stack into [B, E, K]
        expert_outputs = torch.stack(
            [expert(h) for expert in self.experts],
            dim=1
        )
        # Weighted sum of experts: [B, K]
        gate_weights = gate_weights.unsqueeze(-1)  # [B, E, 1]
        logits = (gate_weights * expert_outputs).sum(dim=1)
        return logits

# class Decoder(nn.Module):
#     """
#     Decoder head f_θ concatenates latent embedding z_t and recall embedding r_t,
#     then outputs class logits.
#
#     Following your Core Math:
#       • Inputs: z_t ∈ ℝᵈ, r_t ∈ ℝᵈ
#       • Concatenate → h_t ∈ ℝ²ᵈ
#       • f_θ(h_t) = W₂·ReLU(W₁·h_t + b₁) + b₂ = logits
#     """
#     def __init__(self, latent_dim: int, hidden_dim: int, num_classes: int):
#         super(Decoder, self).__init__()
#         # Linear₁: projects 2d → hidden_dim
#         self.fc1 = nn.Linear(latent_dim * 2, hidden_dim)
#         self.act = nn.ReLU(inplace=True)
#         # Linear₂: projects hidden_dim → num_classes (logits)
#         self.fc2 = nn.Linear(hidden_dim, num_classes)
#
#     def forward(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#           z (torch.Tensor): [B, d], latent embedding from Encoder
#           r (torch.Tensor): [B, d], recall embedding from RecallEngine
#         Returns:
#           logits (torch.Tensor): [B, num_classes]
#         """
#         # 1) concatenate z_t and r_t → [B, 2d]
#         h = torch.cat([z, r], dim=1)
#         # 2) hidden projection + non-linearity
#         h = self.act(self.fc1(h))
#         # 3) final linear to logits
#         logits = self.fc2(h)
#         return logits
