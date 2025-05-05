from typing import Optional, Callable

def reduce(network, topology, rank: int, size: int, data, op: Callable = lambda a, b: a + b, root: int = 0) -> Optional[object]:
    """
    Reduce implementation that works with both Ring and Tree topologies.
    
    This function combines data from all processes using the specified operation
    and returns the result to the root process.
    
    Args:
        network: The network layer for communication
        topology: The topology (ring or tree) for message passing
        rank: The rank of the current process
        size: The total number of processes
        data: The local data to be reduced
        op: The reduction operation (default is addition)
        root: The rank of the root process (default 0)
        
    Returns:
        At root: The result of reducing data from all processes
        At non-root: None
    """
    
    # Implementation for Ring topology
    if isinstance(topology, type) and topology.__name__ == 'RingTopology':
        if rank == root:
            # Root starts with its own data
            result = data
            
            # Receive and reduce data from other nodes
            for _ in range(size - 1):
                recv_result = network.recv()
                if recv_result is None:
                    continue
                
                sender_rank, sender_data = recv_result
                result = op(result, sender_data)
            
            return result
        else:
            # Non-root nodes send their data directly to root
            if network.connect(6000 + root):
                network.send((rank, data))
            return None
    
    # Implementation for Tree topology
    else:  # TreeTopology or any other topology
        # First, receive data from all children and reduce
        result = data
        
        # Collect and reduce data from children
        for _ in range(len(topology.get_children(rank, size))):
            recv_result = network.recv()
            if recv_result is None:
                continue
            
            _, child_data = recv_result
            result = op(result, child_data)
        
        # If not root, send reduced result to parent
        if rank != root:
            parent = topology.get_parent(rank, size)
            if parent is not None and network.connect(6000 + parent):
                network.send((rank, result))
            return None
        
        # At root, return the final reduced result
        return result
    