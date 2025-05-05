from communicator import Communicator
from visualization.visualize import visualize_broadcast
import multiprocessing as mp
import time

def worker(rank: int, size: int, topology_type: str, vis_queue: mp.Queue = None):
    comm = Communicator(rank, size, topology_type)
    vis_steps = [] if vis_queue is not None else None
    
    try:
        if rank != 0:
            time.sleep(1)
        
        data = f"Msg from {rank}" if rank == 0 else None
        result = comm.broadcast_vis(data, vis_steps=vis_steps)
        
        if vis_queue is not None and rank == 0:
            # collect topology data for visualization
            topology_data = {
                r: comm.topology.get_children(r, size)
                for r in range(size)
            }
            vis_queue.put((topology_data, vis_steps))
    
    finally:
        comm.close()
    return result

if __name__ == '__main__':
    size = 4
    topology = 'star'  
    
    vis_queue = mp.Queue()
    processes = []
    
    for rank in range(1, size):
        p = mp.Process(target=worker, args=(rank, size, topology, vis_queue))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    
    p = mp.Process(target=worker, args=(0, size, topology, vis_queue))
    p.start()
    processes.append(p)
    
    for p in processes:
        p.join()
    
    if not vis_queue.empty():
        topology_data, comm_steps = vis_queue.get()
        visualize_broadcast(0, topology_data, comm_steps)