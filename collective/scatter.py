from typing import Optional

def scatter(network, topology, rank: int, size: int, data, root: int = 0) -> Optional[object]:
    """
    Scatter implementation that works with both Ring and Tree topologies.
    
    This function distributes different portions of data from the root node to all other nodes.
    Each node receives a portion of the data array based on its rank.
    
    Args:
        network: The network layer for communication
        topology: The topology (ring or tree) for message passing
        rank: The rank of the current process
        size: The total number of processes
        data: At root, a list/array with one element per process; None elsewhere
        root: The rank of the root process (default 0)
    
    Returns:
        The portion of data destined for this process
    """
    
    # Handle ring topology
    if isinstance(topology, type) and topology.__name__ == 'RingTopology':
        # In ring topology, messages flow in one direction around the ring
        if rank == root:
            # Root sends appropriate chunks to each node
            for i in range(size):
                if i != root:
                    target = i
                    if network.connect(6000 + target):
                        network.send((rank, data[i]))
            # Root keeps its own portion
            return data[root]
        else:
            # Non-root nodes receive their portion from the network
            result = network.recv()
            if result is None:
                return None
            _, my_data = result
            return my_data
            
    # Handle tree topology
    else:  # TreeTopology or any other topology
        if rank == root:
            # Root prepares data for its children
            for child in topology.get_children(rank, size):
                # Prepare data chunks for this subtree
                subtree_data = {}
                # Find all nodes in this child's subtree (including the child)
                queue = [child]
                while queue:
                    node = queue.pop(0)
                    subtree_data[node] = data[node]
                    queue.extend(topology.get_children(node, size))
                
                # Send data to child
                if network.connect(6000 + child):
                    network.send((rank, subtree_data))
            
            # Root keeps its portion
            return data[root]
        else:
            # Receive data from parent
            result = network.recv()
            if result is None:
                return None
            
            _, subtree_data = result
            
            # Extract my portion
            my_data = subtree_data[rank]
            
            # Forward remaining data to my children
            for child in topology.get_children(rank, size):
                # Prepare data for this child's subtree
                child_data = {k: v for k, v in subtree_data.items() if k == child or k in get_subtree_nodes(topology, child, size)}
                
                if network.connect(6000 + child):
                    network.send((rank, child_data))
            
            return my_data

def get_subtree_nodes(topology, node, size):
    """Helper function to get all nodes in a subtree"""
    subtree = []
    queue = [node]
    while queue:
        current = queue.pop(0)
        children = topology.get_children(current, size)
        subtree.extend(children)
        queue.extend(children)
    return subtree
    