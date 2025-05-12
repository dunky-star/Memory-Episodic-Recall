# modules/epinet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder
from .episodic_memory import EpisodicMemory
from .recall_engine import RecallEngine

class EpiNetModel(nn.Module):
    """
    EpiNetModel integrates:
      1) Encoder → latent embedding z_t
      2) RecallEngine/EpisodicMemory → recall embedding r_t
      3) Decoder → class logits
    Computes joint loss:
      L_total = L_task + λ·L_replay
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_classes: int,
        capacity: int,
        decay_rate: float,
        top_k: int,
        lambda_coef: float,
        device: torch.device = None
    ):
        super(EpiNetModel, self).__init__()
        self.device      = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Core modules
        self.encoder       = Encoder(latent_dim)
        self.decoder       = Decoder(latent_dim, hidden_dim, num_classes)
        self.memory        = EpisodicMemory(capacity, latent_dim, decay_rate, self.device)
        self.recall_engine = RecallEngine(top_k)

        # Loss and hyperparameters
        self.criterion   = nn.CrossEntropyLoss()
        self.lambda_coef = lambda_coef

        # Move to device
        self.to(self.device)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        x = x.to(self.device)
        # Encode
        z = self.encoder(x)
        # Recall
        r = self.recall_engine.recall(z, self.memory)
        # Predict
        logits = self.decoder(z, r)
        if y is None:
            return logits

        # Task loss
        y = y.to(self.device)
        loss_task = self.criterion(logits, y)

        # --- Replay loss per Core Math with capacity guard ---
        # Decayed salience: [N]
        salience_mem = self.memory.mem_ctrl.decay(
            self.memory.r0_buffer,
            self.memory.tau_buffer
        )
        # Cosine similarity [B, N]
        cos_sim_mem = F.cosine_similarity(
            z.unsqueeze(1),                # [B,1,d]
            self.memory.c_buffer.unsqueeze(0),  # [1,N,d]
            dim=-1
        )
        # Recall scores [B, N]
        recall_scores = cos_sim_mem * salience_mem.unsqueeze(0)

        # Only compute replay if we have any memories
        N = self.memory.z_buffer.size(0)
        if N > 0:
            # clamp k by current memory size
            k = min(self.recall_engine.top_k, N)
            _, top_idx = torch.topk(
                recall_scores,
                k,
                dim=1
            )
            # Gather for replay
            salience_topk = salience_mem[top_idx]        # [B,K]
            z_topk        = self.memory.z_buffer[top_idx] # [B,K,d]
            y_topk        = self.memory.y_buffer[top_idx] # [B,K]
            # Flatten for batch decode
            B, K, d = z_topk.shape
            z_flat = z_topk.view(B*K, d)
            r_flat = torch.zeros_like(z_flat)
            y_flat = y_topk.view(B*K)
            # Replay logits and per-item losses
            logits_mem = self.decoder(z_flat, r_flat)
            losses_mem = F.cross_entropy(logits_mem, y_flat, reduction='none').view(B, K)
            # Weighted sum for replay loss
            loss_replay = (salience_topk * losses_mem).sum()
            # Total loss with replay
            loss = loss_task + self.lambda_coef * loss_replay
        else:
            # No memories yet, skip replay
            loss = loss_task

        # Update episodic memory
        with torch.no_grad():
            initial_r0 = 1.0
            for zi, yi in zip(z, y):
                # Use z as both embedding & context
                self.memory.add(zi, zi, initial_r0, yi)

        return loss


