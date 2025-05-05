from typing import Optional, List, Dict
import time

def broadcast(network, topology, rank: int, size: int, data) -> Optional[object]:
    if rank != 0:
        result = network.recv()
        if result is None:
            return None
        _, data = result
    
    for child_rank in topology.get_children(rank, size):
        if network.connect(6000 + child_rank):
            network.send((rank, data))
    
    return data


def broadcast_vis(network, topology, rank: int, size: int, data, vis_steps: List[Dict] = None):
    """
    with visualization
    """
    if rank != 0:
        result = network.recv()
        if result is None:
            return None
        src_rank, data = result
        
        if vis_steps is not None:
            vis_steps.append({
                'from': src_rank,
                'to': rank,
                'data': str(data)[:20] + "..." if len(str(data)) > 20 else str(data),
                'time': time.time()
            })
    
    for child_rank in topology.get_children(rank, size):
        if network.connect(6000 + child_rank):
            network.send((rank, data))
            
            if vis_steps is not None:
                vis_steps.append({
                    'from': rank,
                    'to': child_rank,
                    'data': str(data)[:20] + "..." if len(str(data)) > 20 else str(data),
                    'time': time.time()
                })
    
    return data