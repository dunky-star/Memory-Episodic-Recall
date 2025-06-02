
# modules/recall_engine.py

import torch
import torch.nn.functional as F
from .episodic_memory import EpisodicMemory


# 5. RecallEngine Module
# - **top_k**: number of highest‑scoring memories to retrieve
# - **RecallScoreₘ** = cos(z_t, cₘ) · salienceₘ
# - **salienceₘ** = r₀ₘ · exp(–α·(τ_now – τₘ))
# - **Recall embedding**: where the sum is over the Top‑K memories.

class RecallEngine:
    """
    Handles salience decay, scoring, and top-k retrieval from EpisodicMemory.
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
        self.top_k = int(top_k)

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

        task_mask = torch.ones_like(memory.r0_buffer)

        # Compute decayed salience for all stored memories: [N]
        salience = memory.mem_ctrl.decay(memory.r0_buffer, memory.tau_buffer) * task_mask

        if salience.sum() == 0:
            return torch.zeros_like(z_query)

        # Compute cosine similarity: [B, N]
        cos_sim = F.cosine_similarity(
            F.normalize(z_query, dim=-1).unsqueeze(1),  # normalise query
            memory.c_buffer.unsqueeze(0), dim=-1  # memory.c_buffer is already L2-normalised
        ).clamp(min=0)  # keep only non-negative matches

        # Compute recall scores for selection: [B, N]
        recall_scores = cos_sim * salience.unsqueeze(0)
        k = min(self.top_k, recall_scores.size(1))
        # Select Top-K indices by recall score: [B, K]
        _, top_idx = torch.topk(recall_scores, k, dim=1)

        # Gather salience and latent embeddings of selected memories
        salience_topk = salience[top_idx]  # [B, K]
        z_topk = memory.z_buffer[top_idx]  # [B, K, d]

        # Normalize by sum of salience: weights = r_i / Σ_j r_j
        weights = salience_topk / (salience_topk.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted sum to form recall embedding: [B, d]
        recall_emb = (weights.unsqueeze(-1) * z_topk).sum(dim=1)
        return recall_emb