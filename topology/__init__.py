"""
Topology implementations for collective communications
"""

from .star import StarTopology
from .ring import RingTopology
from .tree import TreeTopology

__all__ = ['StarTopology', 'RingTopology', 'TreeTopology']