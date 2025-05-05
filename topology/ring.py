from .base import Topology
from typing import List, Optional
class RingTopology(Topology):
    def get_children(self, rank: int, size: int) -> List[int]:
        return [(rank + 1) % size] if (rank + 1) % size != 0 else []
    
    def get_parent(self, rank: int, size: int) -> Optional[int]:
        return (rank - 1) % size if rank != 0 else None