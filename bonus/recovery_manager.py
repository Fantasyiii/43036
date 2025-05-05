import logging
import threading
import time
import random
from typing import Set, Dict, Callable, List, Any

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bonus/fault_tolerant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RecoveryManager")

class RecoveryManager:
    """Manager for handling node failure detection and recovery"""
    
    def __init__(self, rank: int, size: int):
        self.rank = rank
        self.size = size
        self.failed_nodes = set()  # Set of failed nodes
        self.recovery_in_progress = set()  # Set of nodes currently in recovery
        self.recovery_callbacks = {}  # Callbacks when recovery completes
        self.operation_state = {}  # Store operation state for fault recovery
        
        # Mutex lock for protecting shared state
        self.lock = threading.Lock()
        
        # Record last communication state
        self.last_communication_state = {
            "operation": None,
            "data": None,
            "root": None,
            "step": None,
            "is_complete": False
        }
        
        logger.info(f"Node {self.rank}: Recovery manager initialized")
    
    def node_failed(self, node_rank: int) -> None:
        """Handle node failure"""
        with self.lock:
            if node_rank not in self.failed_nodes:
                logger.warning(f"Node {self.rank}: Node {node_rank} failure detected")
                self.failed_nodes.add(node_rank)
                
                # If operation is in progress, need to reorganize topology and recover
                if self.last_communication_state["operation"] and not self.last_communication_state["is_complete"]:
                    self._initiate_recovery(node_rank)
    
    def node_recovered(self, node_rank: int) -> None:
        """Handle node recovery"""
        with self.lock:
            if node_rank in self.failed_nodes:
                logger.info(f"Node {self.rank}: Node {node_rank} has recovered")
                self.failed_nodes.remove(node_rank)
                
                if node_rank in self.recovery_in_progress:
                    self.recovery_in_progress.remove(node_rank)
                    
                    # Execute recovery completion callback
                    if node_rank in self.recovery_callbacks:
                        callback = self.recovery_callbacks.pop(node_rank)
                        threading.Thread(target=callback).start()
    
    def _initiate_recovery(self, failed_node: int) -> None:
        """Start node failure recovery process"""
        if failed_node in self.recovery_in_progress:
            return  # Already in recovery
        
        self.recovery_in_progress.add(failed_node)
        
        # Record current operation state for recovery
        operation = self.last_communication_state["operation"]
        if operation:
            logger.info(f"Node {self.rank}: Initiating recovery for operation {operation} after node {failed_node} failure")
            
            # Start recovery thread
            threading.Thread(target=self._recovery_procedure, args=(failed_node, operation)).start()
    
    def _recovery_procedure(self, failed_node: int, operation: str) -> None:
        """Execute recovery procedure"""
        # In a real system, this would perform specific recovery steps based on operation type
        # Such as reorganizing topology, reassigning subtasks, or restarting the operation
        
        recovery_time = random.uniform(1.0, 3.0)  # Simulate recovery time
        logger.info(f"Node {self.rank}: Recovery procedure for node {failed_node} started, estimated time: {recovery_time:.2f}s")
        
        time.sleep(recovery_time)  # Simulate recovery process
        
        logger.info(f"Node {self.rank}: Recovery procedure for node {failed_node} completed")
        
        with self.lock:
            # Recovery complete, update state
            if failed_node in self.recovery_in_progress:
                self.recovery_in_progress.remove(failed_node)
    
    def register_operation_state(self, operation: str, data: Any = None, root: int = 0, step: int = 0) -> None:
        """Register operation state for fault recovery"""
        with self.lock:
            self.last_communication_state = {
                "operation": operation,
                "data": data,
                "root": root,
                "step": step,
                "is_complete": False
            }
            
            # Save state copy for recovery
            self.operation_state[operation] = {
                "data": data,
                "root": root,
                "step": step,
                "timestamp": time.time()
            }
            
            logger.debug(f"Node {self.rank}: Registered state for operation {operation}, step {step}")
    
    def mark_operation_complete(self, operation: str) -> None:
        """Mark operation as complete"""
        with self.lock:
            self.last_communication_state["is_complete"] = True
            
            if operation in self.operation_state:
                del self.operation_state[operation]
                
            logger.debug(f"Node {self.rank}: Marked operation {operation} as complete")
    
    def get_active_nodes(self) -> Set[int]:
        """Get set of currently active nodes"""
        with self.lock:
            all_nodes = set(range(self.size))
            return all_nodes - self.failed_nodes
    
    def is_node_failed(self, node_rank: int) -> bool:
        """Check if node has failed"""
        with self.lock:
            return node_rank in self.failed_nodes
    
    def register_recovery_callback(self, node_rank: int, callback: Callable) -> None:
        """Register callback for when recovery completes"""
        with self.lock:
            self.recovery_callbacks[node_rank] = callback
    
    def on_communication_error(self, target_rank: int, error: Exception) -> None:
        """Handle communication error"""
        with self.lock:
            logger.error(f"Node {self.rank}: Communication error with node {target_rank}: {str(error)}")
            
            # Mark node as failed
            if target_rank not in self.failed_nodes:
                self.node_failed(target_rank)
    
    def get_operation_state(self, operation: str) -> Dict:
        """Get operation state for recovery"""
        with self.lock:
            return self.operation_state.get(operation, {}) 