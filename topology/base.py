from abc import ABC, abstractmethod
from typing import List, Optional

class Topology(ABC):
    @abstractmethod
    def get_children(self, rank: int, size: int) -> List[int]:
        """
        get children ranks
        """
        pass
    
    @abstractmethod
    def get_parent(self, rank: int, size: int) -> Optional[int]:
        """
        get parent rank (root node returns None)
        """
        pass