import torch
import time

# 3. MemoryController Module
class MemoryController:
    """
    Computes the biologically-inspired salience decay.
    salience_m = r * exp(–α · (τ_now – τ_m))

    Notation:
      • r      … initial salience score (r₀)
      • τ_m    … timestamp when memory m was stored
      • α      … decay rate (learnable or fixed)
      • salience_m … decayed salience at the current time
    """
    def __init__(self, alpha: float):
        """
        α: decay rate (higher → faster forgetting)
        """
        self.alpha = alpha

    def decay(self, r: torch.Tensor, tau_m: torch.Tensor) -> torch.Tensor:
        """
           Apply decay to a batch of memories.

        Args:
            r      (torch.Tensor): shape [M], initial salience scores for M memories
            tau_m  (torch.Tensor): shape [M], stored timestamps (in seconds) for each memory

        Returns:
            torch.Tensor of shape [M]:
              current salience_m = r * exp(–α · (τ_now – τ_m))
        """
        #  Current time in seconds, on same device as inputs
        tau_now = torch.tensor(time.time(), device=r.device)
        #  Δτ = τ_now – τ_m
        delta_tau = tau_now - tau_m
        #  Apply decay element-wise
        salience_m = r * torch.exp(-self.alpha * delta_tau)
        return salience_m