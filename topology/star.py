from typing import List, Optional
from .base import Topology

class StarTopology(Topology):
    def get_children(self, rank: int, size: int) -> List[int]:
        return list(range(1, size)) if rank == 0 else []
    
    def get_parent(self, rank: int, size: int) -> Optional[int]:
        return 0 if rank != 0 else None