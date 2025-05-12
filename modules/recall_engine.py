
# modules/recall_engine.py

import torch
import torch.nn.functional as F
from .episodic_memory import EpisodicMemory

class RecallEngine:
    """
    Retrieves salient memories based on cosine similarity and decayed salience.

    Given a query embedding z_t and stored memories (z_m, c_m, r0_m, τ_m),
    computes for each memory:
      RecallScore_m = cos(z_t, c_m) * salience_m
    where salience_m = r0_m * exp(-α * (τ_now - τ_m)).
    Selects Top‑K memories by RecallScore, then computes recall embedding:
      r_t = (1 / Σ_i r_i) * Σ_i r_i * z_i,
    where r_i is the decayed salience of the selected memories.
    """
    def __init__(self, top_k: int):
        """
        top_k: number of highest‑scoring memories to retrieve.
        """
        self.top_k = top_k

    def recall(self, z_query: torch.Tensor, memory: EpisodicMemory) -> torch.Tensor:
        """
        Perform memory recall for a batch of query embeddings.

        Args:
            z_query (torch.Tensor): shape [B, d], query latent embeddings z_t.
            memory (EpisodicMemory): contains buffers:
              - c_buffer: [N, d] context embeddings c_m
              - z_buffer: [N, d] latent embeddings z_m
              - r0_buffer: [N]   initial salience r0_m
              - tau_buffer: [N]  timestamps τ_m

        Returns:
            torch.Tensor: shape [B, d], recall embeddings r_t.
        """
        # Guard for empty memory
        if memory.z_buffer.size(0) == 0:
            return torch.zeros_like(z_query)

        # Compute decayed salience for all stored memories: [N]
        salience = memory.mem_ctrl.decay(memory.r0_buffer, memory.tau_buffer)

        # Compute cosine similarity: [B, N]
        cos_sim = F.cosine_similarity(
            z_query.unsqueeze(1),         # [B, 1, d]
            memory.c_buffer.unsqueeze(0),   # [1, N, d]
            dim=-1
        )

        # Compute recall scores for selection: [B, N]
        recall_scores = cos_sim * salience.unsqueeze(0)

        # Select Top-K indices by recall score: [B, K]
        _, top_idx = torch.topk(recall_scores, self.top_k, dim=1)

        # Gather salience and latent embeddings of selected memories
        salience_topk = salience[top_idx]          # [B, K]
        z_topk = memory.z_buffer[top_idx]   # [B, K, d]

        # Normalize by sum of salience: weights = r_i / Σ_j r_j
        weights = salience_topk / (salience_topk.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted sum to form recall embedding: [B, d]
        recall_emb = (weights.unsqueeze(-1) * z_topk).sum(dim=1)
        return recall_emb

