"""
visualize the process of collective communication
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List
from time import sleep

class CommVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.G = nx.DiGraph()
        self.pos = None
        self.current_step = 0
        plt.ion()  
    
    def update_topology(self, topology: Dict[int, List[int]], title: str = ""):
        self.G.clear()
        for src, targets in topology.items():
            for tgt in targets:
                self.G.add_edge(src, tgt)
        
        self.pos = nx.spring_layout(self.G)
        self._draw(title)
    
    def highlight_communication(self, src: int, tgt: int, data: str):
        self.current_step += 1
        title = f"Step {self.current_step}: Rank {src} → Rank {tgt}\nData: {data}"
        
        node_colors = ['lightblue'] * len(self.G.nodes)
        edge_colors = ['gray'] * len(self.G.edges)
        
        path = self._find_path(src, tgt)
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            edge_idx = list(self.G.edges).index((u, v))
            edge_colors[edge_idx] = 'red'
        
        for node in path:
            node_colors[list(self.G.nodes).index(node)] = 'orange'
        
        self._draw(title, node_colors, edge_colors)
        sleep(1)  
    
    def _find_path(self, src: int, tgt: int) -> List[int]:
        try:
            return nx.shortest_path(self.G, source=src, target=tgt)
        except:
            return [src, tgt]  
    
    def _draw(self, title: str = "", node_colors=None, edge_colors=None):
        self.ax.clear()
        
        if node_colors is None:
            node_colors = ['lightblue'] * len(self.G.nodes)
        if edge_colors is None:
            edge_colors = ['gray'] * len(self.G.edges)
        
        nx.draw(self.G, self.pos, ax=self.ax, with_labels=True,
                node_color=node_colors, edge_color=edge_colors,
                node_size=800, arrows=True, arrowstyle='->')
        
        self.ax.set_title(title)
        plt.draw()
        
        # save current image
        filename = f'step_{self.current_step}.png'
        plt.savefig(filename)
        plt.pause(0.1)
    
    def close(self):
        plt.ioff()
        plt.close()

def visualize_broadcast(rank: int, topology_data: Dict[int, List[int]], comm_steps: List[Dict]):
    """
    broadcast process visualization
    """
    vis = CommVisualizer()
    vis.update_topology(topology_data, "Initial Topology")
    
    for step in comm_steps:
        vis.highlight_communication(step['from'], step['to'], step['data'])
    
    if rank == 0:
        plt.show(block=True)
    else:
        vis.close()