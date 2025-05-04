import torch
import torch.nn as nn
from encoder import Encoder
from episodic_memory import EpisodicMemory
from recall_engine import RecallEngine
from decoder import Decoder

class EpiNetModel(nn.Module):
    """
    Orchestrates encoding, episodic storage, recall, and decoding.
    """
    def __init__(
        self,
        latent_dim: int = 128,
        memory_capacity: int = 1000,
        decay_rate: float = 1e-3,
        top_k: int = 5,
        num_classes: int = 10
    ):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.memory = EpisodicMemory(memory_capacity)
        self.recaller = RecallEngine(self.memory, decay_rate, top_k)
        self.decoder = Decoder(latent_dim, latent_dim, num_classes)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        recall_vec = self.recaller.recall(z)
        logits = self.decoder(z, recall_vec)
        return logits, z

    def memorize(self, x, z, y, salience: float):
        self.recaller.store(x, z, y, salience)