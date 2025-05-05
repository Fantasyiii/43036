import socket
import pickle
import time
from typing import Optional, Tuple

class NetworkLayer:
    def __init__(self, rank: int, port: int):
        self.rank = rank
        self.port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.settimeout(10.0) 
        
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)  # 128KB
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)  # 128KB
    
    def start_server(self):
        self._socket.bind(('localhost', self.port))
        self._socket.listen(1)
    
    def connect(self, target_port: int) -> bool:
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
                        return False
            return False
        except (ConnectionRefusedError, socket.timeout):
            return False
    
    def send(self, data) -> bool:
        try:
            self._conn.sendall(pickle.dumps(data))
            return True
        except (ConnectionError, pickle.PicklingError):
            return False
    
    def recv(self) -> Optional[Tuple[int, object]]:
        try:
            conn, _ = self._socket.accept()
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
                            print(f"Node {self.rank}: Socket timeout during data receiving")
                            break
                print(f"Node {self.rank}: Incomplete data received, length: {len(data)}")
                return None
        except (socket.timeout, pickle.UnpicklingError) as e:
            print(f"Node {self.rank}: Error in recv: {type(e).__name__}: {str(e)}")
            return None
    
    def close(self):
        if hasattr(self, '_conn'):
            self._conn.close()
        self._socket.close()