# modules/epinet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder
from .episodic_memory import EpisodicMemory
from .recall_engine import RecallEngine


# 7. EpiNetModel Assembly

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
            device: torch.device | None = None
    ):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, num_classes)
        self.memory = EpisodicMemory(capacity, latent_dim, decay_rate, self.device)
        self.recall_engine = RecallEngine(top_k)

        self.criterion = nn.CrossEntropyLoss()
        self.lambda_coef = lambda_coef
        self.last_replay_loss = 0.0
        self.current_task_id: int = 0

        self.to(self.device)

    # Forward pass
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        x = x.to(self.device)
        z = self.encoder(x)
        self.memory.current_task_id = self.current_task_id
        r = self.recall_engine.recall(z, self.memory)
        logits = self.decoder(z, r)

        # Inference path
        if y is None:
            return logits

        # Supervised task loss
        y = y.to(self.device)
        loss_task = self.criterion(logits, y)

        # # Replay loss
        # salience_mem = self.memory.mem_ctrl.decay(
        #     self.memory.r0_buffer, self.memory.tau_buffer
        # )

        # replay loss: compute decayed salience and mask out current-task memories
        salience_mem = self.memory.mem_ctrl.decay(
            self.memory.r0_buffer,
            self.memory.tau_buffer
        )  # [N]

        salience_mem = salience_mem * (self.memory.t_buffer != self.current_task_id).float()

        cos_sim_mem = F.cosine_similarity(
            z.unsqueeze(1),
            self.memory.c_buffer.unsqueeze(0),
            dim=-1
        )
        recall_scores = cos_sim_mem * salience_mem.unsqueeze(0)

        loss_replay = torch.tensor(0.0, device=self.device)
        N = self.memory.z_buffer.size(0)
        if N:
            k = min(self.recall_engine.top_k, N)

            _, top_idx = torch.topk(recall_scores, k, dim=1)

            sal_topk = salience_mem[top_idx]  # [B,K]
            z_topk = self.memory.z_buffer[top_idx]  # [B,K,d]
            y_topk = self.memory.y_buffer[top_idx]  # [B,K]

            B, K, d = z_topk.shape
            z_flat = z_topk.view(B * K, d)
            r_flat = torch.zeros_like(z_flat)  # zero recall for replay
            y_flat = y_topk.view(B * K)

            logits_mem = self.decoder(z_flat, r_flat)

            losses_mem = F.cross_entropy(
                logits_mem, y_flat, reduction='none'
            ).view(B, K)  # [B,K]

            weight_sum = sal_topk.sum() + 1e-8  # weighted *mean* replay loss
            loss_replay = (sal_topk * losses_mem).sum(dim=1,
                                                      keepdim=True) / weight_sum  # Normalizing (so Train_loss is on the same numeric scale as Val_loss.)
            loss_replay = loss_replay.mean()  # average over batch

        self.last_replay_loss = loss_replay.item()
        loss = loss_task + self.lambda_coef * loss_replay

        # Episodic memory update
        with torch.no_grad():
            for zi, yi, logit_i in zip(z, y, logits):
                per_loss = F.cross_entropy(
                    logit_i.unsqueeze(0),
                    yi.unsqueeze(0),
                    reduction='none'
                ).item()

                # Initial salience = tanh(individual loss)
                init_r0 = torch.tanh(torch.tensor(per_loss)).item()

                # Only store if sufficiently salient
                if init_r0 > 0.05:
                    # use zi both as latent and context
                    # self.memory.add(zi, zi, init_r0, yi)
                    # add new memory with correct task_id
                    self.memory.add(zi, zi, init_r0, yi, task_id=self.current_task_id)

        return loss