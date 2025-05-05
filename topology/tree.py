from typing import List, Optional, Dict
from .base import Topology
import random
import math
from collections import defaultdict

class TreeTopology(Topology):
    def __init__(self):
        self._topology_cache: Dict[int, Dict[str, Dict[int, List[int]]]] = {}
    
    def _generate_tree(self, size: int) -> Dict[str, Dict[int, List[int]]]:
        """
        Generate a tree, one option is a balanced binary tree, other options are also acceptable.

        The root node is default the node with rank 0.

        Returns a dictionary containing children and parent
        """
        if size in self._topology_cache:
            return self._topology_cache[size]
        
        root = 0  # Use node 0 as root
        children = defaultdict(list)
        parent = {root: None}

        # Generate a balanced binary tree
        for i in range(1, size):
            # Parent of node i is (i-1)//2 in a complete binary tree
            p = (i - 1) // 2
            children[p].append(i)
            parent[i] = p
        
        result = {
            'children': dict(children),
            'parent': parent
        }
        self._topology_cache[size] = result
        return result

    def get_children(self, rank: int, size: int) -> List[int]:
        if size < 1:
            return []
        topology = self._generate_tree(size)
        return topology['children'].get(rank, [])

    def get_parent(self, rank: int, size: int) -> Optional[int]:
        if size < 1:
            return None
        topology = self._generate_tree(size)
        return topology['parent'].get(rank)