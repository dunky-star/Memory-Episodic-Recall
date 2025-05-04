import time
import numpy as np
import torch
import torch.nn.functional as F
from memory_controller import MemoryController
from episodic_memory import EpisodicMemory

class RecallEngine:
    """
    Handles salience decay, scoring, and top-k retrieval from EpisodicMemory.
    """
    def __init__(self, memory: EpisodicMemory, decay_rate: float = 1e-3, top_k: int = 5):
        self.memory = memory
        self.decay_rate = decay_rate
        self.top_k = top_k
        self.controller = MemoryController(memory.capacity)

    def recall(self, query_z: torch.Tensor) -> torch.Tensor:
        if not self.memory.memory:
            return torch.zeros_like(query_z)
        now = time.time()
        scores = []
        for entry in self.memory.memory:
            # Time-decayed salience
            delta_t = now - entry['tau']
            entry['r'] = entry['r0'] * np.exp(-self.decay_rate * delta_t)
            # Cosine similarity
            sim = F.cosine_similarity(
                query_z.unsqueeze(0),
                entry['z'].to(query_z.device).unsqueeze(0),
                dim=1
            )[0]
            scores.append((sim * entry['r']).item())

        k = min(self.top_k, len(scores))
        topk_idxs = np.argsort(scores)[-k:]

        # Aggregate top-k embeddings
        recall_vec = torch.stack([
            self.memory.memory[i]['z'].to(query_z.device)
            for i in topk_idxs
        ]).mean(dim=0)
        return recall_vec

    def store(self, x, z, y, salience: float):
        entry = {
            'x': x.detach().cpu(),
            'z': z.detach().cpu(),
            'y': y.detach().cpu(),
            'r0': salience,
            'r': salience,
            'tau': time.time(),
        }
        if self.controller.should_store(salience, self.memory.memory):
            if len(self.memory.memory) < self.memory.capacity:
                self.memory.memory.append(entry)
            else:
                entry['replace_idx'] = self.controller.replace_index(self.memory.memory)
                self.memory.memory.append(entry)