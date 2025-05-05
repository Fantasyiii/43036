from network import NetworkLayer
from topology.ring import RingTopology
from collective.broadcast import broadcast
from topology.star import StarTopology
from topology.tree import TreeTopology
from typing import List, Dict, Callable
from collective.gather import gather
from collective.scatter import scatter
from collective.reduce import reduce
from collective.allreduce import allreduce
from collective.allgather import allgather
from collective.reduce_scatter import reduce_scatter
import re
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("communicator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Communicator")

class Communicator:
    def __init__(self, rank: int, size: int, topology_type='ring'):
        self.rank = rank
        self.size = size
        self.port = 6000 + rank
        
        try:
            # Initialize network
            self.network = NetworkLayer(rank, self.port)
            self.network.start_server()  # Ensure server is started
            
            # Initialize topology
            if topology_type == 'ring':
                self.topology = RingTopology()
            elif topology_type == 'star':
                self.topology = StarTopology()
            elif topology_type == 'tree':
                self.topology = TreeTopology()
            else:
                raise ValueError(f"Unsupported topology type: {topology_type}")
                
            logger.info(f"Communicator initialized for rank {rank} with {topology_type} topology")
            
        except Exception as e:
            logger.error(f"Error initializing communicator on rank {rank}: {str(e)}", exc_info=True)
            raise
    
    def broadcast(self, data, root: int = 0):
        try:
            if self.rank == root and data is None:
                raise ValueError("Root rank must provide data")
            
            logger.info(f"Rank {self.rank}: Broadcasting data from root {root}")
            
            return broadcast(
                network=self.network,
                topology=self.topology,
                rank=self.rank,
                size=self.size,
                data=data if self.rank == root else None
            )
        except Exception as e:
            logger.error(f"Error in broadcast on rank {self.rank}: {str(e)}", exc_info=True)
            raise
    
    def gather(self, data, root: int = 0):
        try:
            logger.info(f"Rank {self.rank}: Gathering data to root {root}")
            
            return gather(
                network=self.network,
                topology=self.topology,
                rank=self.rank,
                size=self.size,
                data=data,
                root=root
            )
        except Exception as e:
            logger.error(f"Error in gather on rank {self.rank}: {str(e)}", exc_info=True)
            raise
    
    def scatter(self, data=None, root: int = 0):
        try:
            logger.info(f"Rank {self.rank}: Scattering data from root {root}")
            
            return scatter(
                network=self.network,
                topology=self.topology,
                rank=self.rank,
                size=self.size,
                data=data if self.rank == root else None,
                root=root
            )
        except Exception as e:
            logger.error(f"Error in scatter on rank {self.rank}: {str(e)}", exc_info=True)
            raise
    
    def reduce(self, data, op=lambda a, b: a + b, root: int = 0):
        try:
            logger.info(f"Rank {self.rank}: Reducing data to root {root}")
            
            return reduce(
                network=self.network,
                topology=self.topology,
                rank=self.rank,
                size=self.size,
                data=data,
                op=op,
                root=root
            )
        except Exception as e:
            logger.error(f"Error in reduce on rank {self.rank}: {str(e)}", exc_info=True)
            raise
    
    def allreduce(self, data, op=lambda a, b: a + b):
        try:
            logger.info(f"Rank {self.rank}: Performing allreduce")
            
            return allreduce(
                network=self.network,
                topology=self.topology,
                rank=self.rank,
                size=self.size,
                data=data,
                op=op
            )
        except Exception as e:
            logger.error(f"Error in allreduce on rank {self.rank}: {str(e)}", exc_info=True)
            raise
    
    def allgather(self, data):
        try:
            logger.info(f"Rank {self.rank}: Performing allgather")
            
            return allgather(
                network=self.network,
                topology=self.topology,
                rank=self.rank,
                size=self.size,
                data=data
            )
        except Exception as e:
            logger.error(f"Error in allgather on rank {self.rank}: {str(e)}", exc_info=True)
            raise
    
    def reduce_scatter(self, data, op=lambda a, b: a + b):
        try:
            logger.info(f"Rank {self.rank}: Performing reduce-scatter")
            
            return reduce_scatter(
                network=self.network,
                topology=self.topology,
                rank=self.rank,
                size=self.size,
                data=data,
                op=op
            )
        except Exception as e:
            logger.error(f"Error in reduce_scatter on rank {self.rank}: {str(e)}", exc_info=True)
            raise
    
    def close(self):
        try:
            self.network.close()
            logger.info(f"Rank {self.rank}: Communicator closed")
        except Exception as e:
            logger.error(f"Error closing communicator on rank {self.rank}: {str(e)}", exc_info=True)