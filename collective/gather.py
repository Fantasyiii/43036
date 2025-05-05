from typing import Optional, List

def gather(network, topology, rank: int, size: int, data, root: int = 0) -> Optional[object]:
    """
    Gather implementation that works with both Ring and Tree topologies.
    
    This function collects data from all processes and gathers it at the root process.
    
    Args:
        network: The network layer for communication
        topology: The topology (ring or tree) for message passing
        rank: The rank of the current process
        size: The total number of processes
        data: The local data to be gathered
        root: The rank of the root process (default 0)
    
    Returns:
        At the root: A list containing the gathered data from all processes
        At non-root processes: None or the local data depending on implementation
    """
    
    # Implementation for Ring topology
    if isinstance(topology, type) and topology.__name__ == 'RingTopology':
        # In ring topology, each node passes data around the ring
        if rank == root:
            # Root starts with its own data
            gathered_data = [None] * size
            gathered_data[root] = data
            
            # Root receives data from other nodes
            for _ in range(size - 1):
                result = network.recv()
                if result is None:
                    continue
                sender_rank, sender_data = result
                gathered_data[sender_rank] = sender_data
            
            return gathered_data
        else:
            # Non-root nodes send their data directly to root
            if network.connect(6000 + root):
                network.send((rank, data))
            return None
    
    # Implementation for Tree topology
    else:  # TreeTopology or any other topology
        # Leaf nodes send data up to their parent
        # Inner nodes collect data from children, add their own, then send up
        
        # First, collect data from children
        gathered_data = {}
        gathered_data[rank] = data
        
        # Receive data from all children
        for _ in range(len(topology.get_children(rank, size))):
            result = network.recv()
            if result is None:
                continue
            _, child_data = result
            gathered_data.update(child_data)
        
        # If not root, send gathered data to parent
        if rank != root:
            parent = topology.get_parent(rank, size)
            if parent is not None and network.connect(6000 + parent):
                network.send((rank, gathered_data))
            return None
        
        # At root, convert the dictionary to a list
        if rank == root:
            result_list = [None] * size
            for i, data in gathered_data.items():
                result_list[i] = data
            return result_list