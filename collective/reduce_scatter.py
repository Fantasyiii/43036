from typing import Optional, Callable, List
from .reduce import reduce
from .scatter import scatter

def reduce_scatter(network, topology, rank: int, size: int, data, op: Callable = lambda a, b: a + b) -> Optional[object]:
    """
    Reduce-scatter implementation that works with both Ring and Tree topologies.
    
    This function combines data from all processes using the specified operation
    and then scatters the results, giving each process a portion of the reduced data.
    
    Each process must provide an input array with 'size' elements. Element i of the 
    result will be sent to process i.
    
    Args:
        network: The network layer for communication
        topology: The topology (ring or tree) for message passing
        rank: The rank of the current process
        size: The total number of processes
        data: The local data to be reduced and scattered (list with size elements)
        op: The reduction operation (default is addition)
        
    Returns:
        The portion of the reduced data for this process
    """
    
    # Check that data has the right size
    if len(data) != size:
        raise ValueError(f"Data length ({len(data)}) must match process count ({size})")
    
    # Implementation for Ring topology
    if isinstance(topology, type) and topology.__name__ == 'RingTopology':
        # For Ring topology, we can do a direct implementation
        # Each process reduces data for a specific position and keeps its own
        
        # First, each process starts with its own contribution for its position
        my_result = data[rank]
        
        # For each other process
        for i in range(1, size):
            # Determine source and destination for this step
            src = (rank - i) % size
            dst = (rank + 1) % size
            
            # Send my current data to dst
            if network.connect(6000 + dst):
                network.send((rank, my_result))
            
            # Receive from src and reduce
            result = network.recv()
            if result is not None:
                _, received_data = result
                my_result = op(my_result, received_data)
        
        return my_result
    
    # Implementation for Tree topology
    else:  # TreeTopology or any other topology
        # For tree topology, we can implement this in two phases:
        # 1. Each process reduces data for all positions
        # 2. Scatter the reduced results
        
        root = 0
        
        # Step 1: Each process reduces its input locally
        local_reduced = [None] * size
        for i in range(size):
            local_reduced[i] = data[i]
        
        # Step 2: Perform reduce to root
        if rank == root:
            # Root receives and reduces data from all other processes
            for p in range(1, size):
                result = network.recv()
                if result is None:
                    continue
                
                sender_rank, sender_data = result
                # Combine received data with local data
                for i in range(size):
                    local_reduced[i] = op(local_reduced[i], sender_data[i])
            
            # Step 3: Scatter the reduced results
            # Root keeps its portion and sends the rest
            for p in range(1, size):
                if network.connect(6000 + p):
                    network.send((rank, local_reduced[p]))
            
            return local_reduced[root]
        else:
            # Non-root processes send their data to root
            if network.connect(6000 + root):
                network.send((rank, local_reduced))
            
            # Then receive their portion of the reduced result
            result = network.recv()
            if result is None:
                return None
            
            _, my_portion = result
            return my_portion
    