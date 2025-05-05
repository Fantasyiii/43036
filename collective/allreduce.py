from typing import Optional, Callable
from .reduce import reduce
from .broadcast import broadcast

def allreduce(network, topology, rank: int, size: int, data, op: Callable = lambda a, b: a + b) -> Optional[object]:
    """
    Allreduce implementation that works with both Ring and Tree topologies.
    
    This function combines data from all processes using the specified operation
    and distributes the result back to all processes.
    
    It can be implemented as a reduce operation followed by a broadcast.
    
    Args:
        network: The network layer for communication
        topology: The topology (ring or tree) for message passing
        rank: The rank of the current process
        size: The total number of processes
        data: The local data to be reduced
        op: The reduction operation (default is addition)
        
    Returns:
        The result of reducing data from all processes
    """
    
    # Implementation for Ring topology
    if isinstance(topology, type) and topology.__name__ == 'RingTopology':
        # For ring topology, we'll implement it in two phases:
        # 1. Reduce to root (rank 0)
        # 2. Broadcast from root to all
        
        root = 0
        
        # Phase 1: Reduce to root
        reduced_data = reduce(network, topology, rank, size, data, op, root)
        
        # Phase 2: Broadcast from root
        if rank == root:
            # At root, broadcast the reduced result
            return broadcast(network, topology, rank, size, reduced_data)
        else:
            # At non-root, receive the broadcast
            return broadcast(network, topology, rank, size, None)
    
    # Implementation for Tree topology
    else:  # TreeTopology or any other topology
        # Similar approach for tree topology
        root = 0
        
        # Phase 1: Reduce to root
        reduced_data = reduce(network, topology, rank, size, data, op, root)
        
        # Phase 2: Broadcast from root
        if rank == root:
            # At root, broadcast the reduced result
            return broadcast(network, topology, rank, size, reduced_data)
        else:
            # At non-root, receive the broadcast
            return broadcast(network, topology, rank, size, None)
