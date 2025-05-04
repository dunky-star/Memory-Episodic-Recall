import time
import torch

class EpisodicMemory:
    """
    Stores raw episodes (x, z, y, r0, r, tau) without retrieval logic.
    """
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memory = []  # list of dicts: {x, z, y, r0, r, tau}

    def add(self, entry: dict):
        if len(self.memory) < self.capacity:
            self.memory.append(entry)
        else:
            idx = entry.pop('replace_idx', None)
            if idx is not None:
                self.memory[idx] = entry