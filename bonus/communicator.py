import sys
import os

# Add parent directory to path for importing original modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bonus.network import FaultTolerantNetworkLayer
from bonus.recovery_manager import RecoveryManager
from bonus.topology.tree import FaultTolerantTreeTopology
from typing import List, Dict, Callable, Any, Set, Optional
import time
import threading
import logging
import random

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bonus/fault_tolerant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FT-Communicator")

class FaultTolerantCommunicator:
    """Communicator with fault tolerance capabilities"""
    
    def __init__(self, rank: int, size: int, topology_type='tree'):
        self.rank = rank
        self.size = size
        self.port = 7000 + rank  # Using different port from original to avoid conflicts
        
        try:
            # Initialize recovery manager
            self.recovery_manager = RecoveryManager(rank, size)
            
            # Initialize network layer
            self.network = FaultTolerantNetworkLayer(rank, self.port, size, self.recovery_manager)
            self.network.start_server()
            
            # Initialize topology
            if topology_type == 'tree':
                self.topology = FaultTolerantTreeTopology()
            else:
                raise ValueError(f"Unsupported topology type: {topology_type}")
            
            # State management
            self.operation_in_progress = False
            self.operation_lock = threading.Lock()
            
            logger.info(f"Node {rank}: Fault-tolerant communicator initialized with {topology_type} topology")
        
        except Exception as e:
            logger.error(f"Node {rank}: Error initializing communicator: {str(e)}", exc_info=True)
            raise
    
    def broadcast(self, data, root: int = 0):
        """Fault-tolerant broadcast operation"""
        return self._execute_collective_operation("broadcast", self._broadcast_impl, data, root=root)
    
    def _broadcast_impl(self, data, root: int = 0):
        """Broadcast implementation with fault tolerance"""
        try:
            # Register operation state
            self.recovery_manager.register_operation_state("broadcast", data, root)
            
            if self.rank == root:
                if data is None:
                    raise ValueError("Root rank must provide data")
                result = data
            else:
                result = None
            
            # Get active nodes
            active_nodes = self.network.get_active_nodes()
            
            if root not in active_nodes:
                logger.error(f"Node {self.rank}: Broadcast root {root} is not active")
                return None
            
            if self.rank == root:
                # Root node sends data to all active child nodes
                children = self.topology.get_children(self.rank, self.size, active_nodes)
                for child in children:
                    if self.network.connect(self.port - self.rank + child):
                        self.network.send(data)
                    else:
                        logger.warning(f"Node {self.rank}: Failed to connect to child {child} during broadcast")
            else:
                # Non-root nodes receive data from parent and forward to children
                parent = self.topology.get_parent(self.rank, self.size, active_nodes)
                
                if parent is not None:
                    received = self.network.recv()
                    if received is not None:
                        try:
                            # Safely unpack the data
                            if isinstance(received, tuple) and len(received) == 2:
                                _, result = received
                            else:
                                logger.warning(f"Node {self.rank}: Received unexpected data format during broadcast: {received}")
                                result = received  # Try to use the received data directly
                            
                            # Forward data to child nodes
                            children = self.topology.get_children(self.rank, self.size, active_nodes)
                            for child in children:
                                if self.network.connect(self.port - self.rank + child):
                                    self.network.send(result)
                                else:
                                    logger.warning(f"Node {self.rank}: Failed to connect to child {child} during broadcast")
                        except Exception as e:
                            logger.error(f"Node {self.rank}: Error processing received data: {str(e)}")
                            # Try to use received data even when error occurs
                            result = received
                else:
                    logger.error(f"Node {self.rank}: No active parent for broadcast")
            
            # Mark operation as complete
            self.recovery_manager.mark_operation_complete("broadcast")
            
            return result
        
        except Exception as e:
            logger.error(f"Node {self.rank}: Error in broadcast: {str(e)}", exc_info=True)
            raise
    
    def reduce(self, data, op=lambda a, b: a + b, root: int = 0):
        """Fault-tolerant reduce operation"""
        return self._execute_collective_operation("reduce", self._reduce_impl, data, op=op, root=root)
    
    def _reduce_impl(self, data, op=lambda a, b: a + b, root: int = 0):
        """Reduce implementation with fault tolerance"""
        try:
            # Register operation state
            self.recovery_manager.register_operation_state("reduce", data, root)
            
            # Get active nodes
            active_nodes = self.network.get_active_nodes()
            
            if root not in active_nodes:
                logger.error(f"Node {self.rank}: Reduce root {root} is not active")
                return None
            
            # Local result
            local_result = data
            
            # Collect and merge data from child nodes
            children = self.topology.get_children(self.rank, self.size, active_nodes)
            for child in children:
                received = self.network.recv()
                if received is not None:
                    try:
                        # Safely unpack the data
                        if isinstance(received, tuple) and len(received) == 2:
                            child_rank, child_data = received
                            local_result = op(local_result, child_data)
                        else:
                            logger.warning(f"Node {self.rank}: Received unexpected data format during reduce: {received}")
                            # Try to use received data directly
                            try:
                                local_result = op(local_result, received)
                            except:
                                logger.error(f"Node {self.rank}: Cannot apply reduction operation to received data")
                    except Exception as e:
                        logger.error(f"Node {self.rank}: Error processing received data: {str(e)}")
                else:
                    logger.warning(f"Node {self.rank}: Failed to receive data from child during reduce")
            
            # If not root, send result to parent
            if self.rank != root:
                parent = self.topology.get_parent(self.rank, self.size, active_nodes)
                if parent is not None:
                    if self.network.connect(self.port - self.rank + parent):
                        self.network.send(local_result)
                    else:
                        logger.warning(f"Node {self.rank}: Failed to connect to parent {parent} during reduce")
            
            # Mark operation as complete
            self.recovery_manager.mark_operation_complete("reduce")
            
            # Root returns final result, non-root returns None
            return local_result if self.rank == root else None
        
        except Exception as e:
            logger.error(f"Node {self.rank}: Error in reduce: {str(e)}", exc_info=True)
            raise
    
    def _execute_collective_operation(self, operation_name, impl_func, data, **kwargs):
        """Execute collective operation with fault tolerance mechanism"""
        with self.operation_lock:
            self.operation_in_progress = True
            
            # Randomly trigger simulated node failure (for testing only)
            if random.random() < 0.05 and operation_name != "fault_injection":  # 5% chance of failure
                logger.warning(f"Node {self.rank}: Simulating fault during {operation_name}")
                time.sleep(1)  # Delay to allow fault detection
                
                # Recover operation
                return self._recover_from_failure(operation_name, impl_func, data, **kwargs)
            
            try:
                # Normal operation execution
                result = impl_func(data, **kwargs)
                self.operation_in_progress = False
                return result
            
            except Exception as e:
                logger.error(f"Node {self.rank}: Error executing {operation_name}: {str(e)}", exc_info=True)
                
                # Handle error, attempt recovery
                return self._recover_from_failure(operation_name, impl_func, data, **kwargs)
    
    def _recover_from_failure(self, operation_name, impl_func, data, **kwargs):
        """Recover from operation failure"""
        logger.info(f"Node {self.rank}: Attempting to recover from failure in {operation_name}")
        
        # Wait briefly to allow fault detection mechanism to update
        time.sleep(2.0)
        
        try:
            # Retry operation
            result = impl_func(data, **kwargs)
            logger.info(f"Node {self.rank}: Successfully recovered {operation_name}")
            return result
        
        except Exception as e:
            logger.error(f"Node {self.rank}: Recovery failed for {operation_name}: {str(e)}", exc_info=True)
            return None
        
        finally:
            self.operation_in_progress = False
    
    def inject_fault(self):
        """Inject fault for testing purposes"""
        logger.warning(f"Node {self.rank}: Injecting fault")
        
        # Simulate node failure
        if random.random() < 0.5:
            # Simulate crash
            os._exit(1)
        else:
            # Simulate network failure
            self.network.heartbeat_running = False
            time.sleep(10.0)
            
            # Recover network
            self.network.heartbeat_running = True
            self.network.start_heartbeat()
            
            logger.info(f"Node {self.rank}: Network recovered after simulated fault")
    
    def close(self):
        """Close the communicator and related resources"""
        try:
            self.network.close()
            logger.info(f"Node {self.rank}: Communicator closed")
        except Exception as e:
            logger.error(f"Node {self.rank}: Error closing communicator: {str(e)}") 