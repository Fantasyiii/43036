import socket
import time
import multiprocessing as mp

class Barrier:
    def __init__(self, rank, size, base_port=7000):
        self.rank = rank
        self.size = size
        self.base_port = base_port
        
        self.control_port = base_port
        self.node_ports = [base_port + 1 + i for i in range(size)]
    
    def wait(self):
        if self.rank == 0:
            control_socket = socket.socket()
            control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            control_socket.bind(('localhost', self.control_port))
            control_socket.listen(self.size - 1)
            
            connections = []
            for _ in range(self.size - 1):
                conn, _ = control_socket.accept()
                connections.append(conn)
            
            for conn in connections:
                conn.sendall(b'continue')
                conn.close()
            
            control_socket.close()
        else:
            while True:
                try:
                    s = socket.socket()
                    s.connect(('localhost', self.control_port))
                    s.recv(8)  # wait for continue signal
                    s.close()
                    break
                except (ConnectionRefusedError, ConnectionError):
                    time.sleep(0.1)
                finally:
                    if 's' in locals():
                        s.close()