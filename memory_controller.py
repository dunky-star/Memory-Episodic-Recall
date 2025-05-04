import numpy as np

class MemoryController:
    """
    Decides whether to admit or replace entries in the memory buffer.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity

    def should_store(self, salience: float, buffer: list) -> bool:
        if len(buffer) < self.capacity:
            return True
        min_r = min(entry['r'] for entry in buffer)
        return salience > min_r

    def replace_index(self, buffer: list) -> int:
        saliences = [entry['r'] for entry in buffer]
        return int(np.argmin(saliences))