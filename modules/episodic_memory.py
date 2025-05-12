import time
import torch

from modules.memory_controller import MemoryController


class EpisodicMemory:
    """
    Fixed‑capacity buffer storing tuples, m = (z, c, r₀, τₘ, yₘ).
    admitting only the most salient memories over time.
    Admission & eviction are driven by current salience decay:
      • Admit new memory if under capacity,
      • Otherwise evict the memory with lowest decayed salience.
    """
    def __init__(
            self,
            capacity: int,
            latent_dim: int,
            decay_rate: float,
            device: torch.device
    ):
        """
        Args:
            capacity (int): maximum number of memories
            latent_dim (int): dimensionality of the latent space
            decay_rate (float): decay rate for salience decay
            device (torch.device): device to store the memory
        """
        self.capacity = capacity
        self.device = device

        # Memory buffer (initially empty)
        self.z_buffer = torch.empty((0, latent_dim), device=device) # z: [N × d] latent embeddings z
        self.c_buffer = torch.empty((0, latent_dim), device=device) # c: [N × d] context vectors
        self.r0_buffer = torch.empty((0,), device=device)  # r0: [N] initial salience score r₀
        self.tau_buffer = torch.empty((0,), device=device) # timestamps τₘ when stored
        self.y_buffer = torch.empty((0,), dtype=torch.long, device=device) # labels yₘ

        # Reuse your MemoryController for decay
        self.mem_ctrl = MemoryController(decay_rate)

    def add(
            self,
            z: torch.Tensor,
            c: torch.Tensor,
            r0: float,
            y: torch.Tensor
    ):
        """
        Try to admit new memory (z,c,r0,τ,y).
        If at capacity, evict the lowest‐decayed‐salience memory.
        """
        # Preparing tensors for concatenation
        z = z.detach().to(self.device).view(1, -1)
        c = c.detach().to(self.device).view(1, -1)
        r0 = torch.tensor([r0], device=self.device)
        y = y.detach().to(self.device).view(1)
        tau = torch.tensor([time.time()], device=self.device)

        # If under capacity, just append/admit
        if self.z_buffer.shape[0] < self.capacity:
            self._append(z, c, r0, tau, y)
            return

        # Otherwise compute decayed salience of existing memories
        s_existing = self.mem_ctrl.decay(self.r0_buffer, self.tau_buffer)

        # If this new memory isn't more salience than the least one, skip
        if r0 <= s_existing.min():
            return

        # Else evict the lowest‐salience and replace it
        idx = torch.argmin(s_existing).item()
        self._replace(idx, z, c, r0, tau, y)

    def _append(self, z, c, r0, tau, y):
        """Add a new memory at the end of each buffer."""
        self.z_buffer = torch.cat([self.z_buffer, z], dim=0)
        self.c_buffer = torch.cat([self.c_buffer, c], dim=0)
        self.r0_buffer = torch.cat([self.r0_buffer, r0], dim=0)
        self.tau_buffer = torch.cat([self.tau_buffer, tau], dim=0)
        self.y_buffer = torch.cat([self.y_buffer, y], dim=0)


    def _replace(self, idx, z, c, r0, tau, y):
        """Overwrite the memory at index `idx` with the new one."""
        self.z_buffer[idx] = z
        self.c_buffer[idx] = c
        self.r0_buffer[idx] = r0
        self.tau_buffer[idx] = tau
        self.y_buffer[idx] = y

    def clear(self):
        """Reset all buffers to empty."""
        self.__init__(self.capacity, self.z_buffer.size(1), self.mem_ctrl.alpha, self.device)
