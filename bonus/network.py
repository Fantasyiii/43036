import socket
import pickle
import time
import threading
import logging
from typing import Optional, Tuple, Dict, List, Set

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bonus/fault_tolerant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FT-Network")

class FaultTolerantNetworkLayer:
    def __init__(self, rank: int, port: int, size: int, recovery_manager=None):
        self.rank = rank
        self.port = port
        self.size = size
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.settimeout(10.0)
        
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)  # 128KB
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)  # 128KB
        
        # Fault detection and recovery
        self.active_nodes = set(range(size))  # All active nodes
        self.failed_nodes = set()  # Failed nodes
        self.recovery_manager = recovery_manager  # Recovery manager reference
        
        # Heartbeat mechanism
        self.heartbeat_interval = 2.0  # Heartbeat interval (seconds)
        self.heartbeat_timeout = 5.0  # Heartbeat timeout (seconds)
        self.last_heartbeat = {}  # Record last heartbeat time for each node
        self.heartbeat_running = False
        self.heartbeat_thread = None
        
        # Separate socket for receiving and processing heartbeats
        self.heartbeat_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.heartbeat_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    def start_server(self):
        self._socket.bind(('localhost', self.port))
        self._socket.listen(10)  # Increase listen queue size
        
        # Start heartbeat mechanism
        self.heartbeat_socket.bind(('localhost', self.port + 10000))  # Use different port
        self.start_heartbeat()
        
        logger.info(f"Node {self.rank}: Server and heartbeat mechanism started")
    
    def start_heartbeat(self):
        """Start heartbeat mechanism"""
        if not self.heartbeat_running:
            self.heartbeat_running = True
            
            # Start heartbeat sender thread
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_sender)
            self.heartbeat_thread.daemon = True
            self.heartbeat_thread.start()
            
            # Start heartbeat receiver thread
            self.heartbeat_receiver_thread = threading.Thread(target=self._heartbeat_receiver)
            self.heartbeat_receiver_thread.daemon = True
            self.heartbeat_receiver_thread.start()
            
            # Start fault detection thread
            self.failure_detector_thread = threading.Thread(target=self._failure_detector)
            self.failure_detector_thread.daemon = True
            self.failure_detector_thread.start()
    
    def _heartbeat_sender(self):
        """Send heartbeats to all other nodes"""
        while self.heartbeat_running:
            for node in range(self.size):
                if node != self.rank and node not in self.failed_nodes:
                    try:
                        heartbeat_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        heartbeat_socket.sendto(
                            pickle.dumps((self.rank, "heartbeat")),
                            ('localhost', 10000 + node + self.port)
                        )
                        heartbeat_socket.close()
                    except Exception as e:
                        logger.warning(f"Node {self.rank}: Error sending heartbeat to node {node}: {str(e)}")
            time.sleep(self.heartbeat_interval)
    
    def _heartbeat_receiver(self):
        """Receive heartbeats from other nodes"""
        self.heartbeat_socket.settimeout(0.5)  # Set short timeout to handle close requests
        
        while self.heartbeat_running:
            try:
                data, _ = self.heartbeat_socket.recvfrom(1024)
                sender, message = pickle.loads(data)
                
                if message == "heartbeat":
                    self.last_heartbeat[sender] = time.time()
                    
                    # If previously marked failed node has recovered, mark as recovered
                    if sender in self.failed_nodes:
                        logger.info(f"Node {self.rank}: Detected recovery of node {sender}")
                        self.failed_nodes.remove(sender)
                        self.active_nodes.add(sender)
                        
                        # Notify recovery manager
                        if self.recovery_manager:
                            self.recovery_manager.node_recovered(sender)
            
            except socket.timeout:
                continue  # Timeout, continue loop
            except Exception as e:
                logger.warning(f"Node {self.rank}: Error receiving heartbeat: {str(e)}")
    
    def _failure_detector(self):
        """Detect node failures"""
        while self.heartbeat_running:
            current_time = time.time()
            
            for node in range(self.size):
                if node != self.rank and node not in self.failed_nodes:
                    # If no heartbeat received beyond timeout, consider node failed
                    if node in self.last_heartbeat and current_time - self.last_heartbeat[node] > self.heartbeat_timeout:
                        logger.warning(f"Node {self.rank}: Detected failure of node {node}")
                        self.failed_nodes.add(node)
                        self.active_nodes.remove(node)
                        
                        # Notify recovery manager
                        if self.recovery_manager:
                            self.recovery_manager.node_failed(node)
            
            time.sleep(1.0)  # Check once per second
    
    def connect(self, target_port: int) -> bool:
        """Connect to target node, with retry logic"""
        try:
            self._conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            max_attempts = 3
            retry_delay = 0.5
            for attempt in range(max_attempts):
                try:
                    self._conn.connect(('localhost', target_port))
                    return True
                except (ConnectionRefusedError, socket.timeout):
                    if attempt < max_attempts - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Connection failed, node may have failed
                        target_rank = target_port - 6000
                        if target_rank >= 0 and target_rank < self.size:
                            logger.warning(f"Node {self.rank}: Connection to node {target_rank} failed, marking as failed")
                            self.failed_nodes.add(target_rank)
                            self.active_nodes.discard(target_rank)
                            
                            # Notify recovery manager
                            if self.recovery_manager:
                                self.recovery_manager.node_failed(target_rank)
                        return False
            return False
        except (ConnectionRefusedError, socket.timeout):
            return False
    
    def send(self, data) -> bool:
        """Send data with error handling"""
        try:
            self._conn.sendall(pickle.dumps(data))
            return True
        except (ConnectionError, pickle.PicklingError) as e:
            logger.error(f"Node {self.rank}: Error sending data: {str(e)}")
            return False
    
    def recv(self) -> Optional[Tuple[int, object]]:
        """Receive data with error handling and retry logic"""
        try:
            conn, addr = self._socket.accept()
            conn.settimeout(20.0)
            with conn:
                data = b""
                chunk = conn.recv(131072)
                while chunk:
                    data += chunk
                    try:
                        return pickle.loads(data)
                    except (pickle.UnpicklingError, EOFError):
                        try:
                            chunk = conn.recv(131072)
                        except socket.timeout:
                            logger.warning(f"Node {self.rank}: Socket timeout during data receiving")
                            break
                logger.warning(f"Node {self.rank}: Incomplete data received, length: {len(data)}")
                return None
        except (socket.timeout, pickle.UnpicklingError) as e:
            logger.error(f"Node {self.rank}: Error in recv: {type(e).__name__}: {str(e)}")
            return None
    
    def get_active_nodes(self) -> Set[int]:
        """Return set of currently active nodes"""
        return self.active_nodes
    
    def get_failed_nodes(self) -> Set[int]:
        """Return set of currently failed nodes"""
        return self.failed_nodes
    
    def close(self):
        """Close network connections"""
        # Stop heartbeat mechanism
        self.heartbeat_running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1.0)
        if hasattr(self, 'heartbeat_receiver_thread'):
            self.heartbeat_receiver_thread.join(timeout=1.0)
        if hasattr(self, 'failure_detector_thread'):
            self.failure_detector_thread.join(timeout=1.0)
        
        # Close sockets
        if hasattr(self, '_conn'):
            self._conn.close()
        self._socket.close()
        self.heartbeat_socket.close()
        
        logger.info(f"Node {self.rank}: Network layer closed") 