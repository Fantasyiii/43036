from typing import Optional, List
from .gather import gather
from .broadcast import broadcast

def allgather(network, topology, rank: int, size: int, data) -> Optional[object]:
    """
    Allgather implementation that works with both Ring and Tree topologies.
    
    This function collects data from all processes and distributes the result back to all processes.
    It can be implemented as a gather followed by a broadcast.
    
    Args:
        network: The network layer for communication
        topology: The topology (ring or tree) for message passing
        rank: The rank of the current process
        size: The total number of processes
        data: The local data to be gathered and distributed
        
    Returns:
        A list containing the gathered data from all processes
    """
    
    # Implementation for Ring topology
    if isinstance(topology, type) and topology.__name__ == 'RingTopology':
        # In ring topology, first gather data to a root, then broadcast it
        # For simplicity, we'll use rank 0 as the root
        root = 0
        
        # First perform a gather to the root
        gathered_data = None
        if rank == root:
            # Root collects data from all nodes
            all_data = [None] * size
            all_data[root] = data
            
            # Receive from each other node
            for _ in range(size - 1):
                result = network.recv()
                if result is None:
                    continue
                sender_rank, sender_data = result
                all_data[sender_rank] = sender_data
                
            gathered_data = all_data
        else:
            # Send data to root
            if network.connect(6000 + root):
                network.send((rank, data))
        
        # Now broadcast the gathered data from root to all
        if rank == root:
            # Root sends to its next node in the ring
            next_node = (rank + 1) % size
            if network.connect(6000 + next_node):
                network.send((rank, gathered_data))
            return gathered_data
        else:
            # Receive from previous node and pass along
            result = network.recv()
            if result is None:
                return None
                
            sender_rank, all_data = result
            
            # Forward to next node in the ring (except the last one)
            next_node = (rank + 1) % size
            if next_node != root and network.connect(6000 + next_node):
                network.send((rank, all_data))
                
            return all_data
    
    # Implementation for Tree topology or other topologies
    else:
        # For tree topology, we can implement allgather as gather followed by broadcast
        # First perform a gather to root (default 0)
        root = 0
        gathered_data = gather(network, topology, rank, size, data, root)
        
        # Then broadcast the result from root to all nodes
        if rank == root:
            # At root, broadcast the gathered data
            result = broadcast(network, topology, rank, size, gathered_data)
            return result
        else:
            # At non-root, receive the broadcast
            result = broadcast(network, topology, rank, size, None)
            return result
