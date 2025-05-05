from topology.tree import MSTTopology
import matplotlib.pyplot as plt
import networkx as nx

def visualize_multiple_msts():
    """Visualize MST topologies of different sizes"""
    sizes = [4, 8, 12]
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, size in enumerate(sizes):
        topo = MSTTopology()
        mst_data = topo._generate_mst(size)
        
        G = nx.DiGraph()
        for u, children in mst_data['children'].items():
            for v in children:
                G.add_edge(u, v)
        
        # Draw on the corresponding subplot
        ax = axes[i]
        pos = nx.kamada_kawai_layout(G)  # Use a better layout algorithm
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', 
                node_size=500, arrows=True, arrowsize=15)
        ax.set_title(f"MST Topology (Nodes={size})")
    
    plt.tight_layout()
    plt.savefig('multiple_mst_topologies.png')
    plt.show()

def visualize_mst(size: int):
    """Visualize the generated MST"""
    topo = MSTTopology()
    topo._generate_mst(size)
    
    G = nx.DiGraph()
    for rank in range(size):
        for child in topo.get_children(rank, size):
            G.add_edge(rank, child)
    
    pos = nx.spring_layout(G)
    pos = nx.kamada_kawai_layout(G)  # Use a better layout algorithm
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, arrows=True)
    plt.title(f"MST Topology (Size={size})")
    plt.savefig(f'mst_topology_size_{size}.png')
    plt.close()
    import os
    print(f"Current working directory: {os.getcwd()}")

if __name__ == '__main__':
    # Visualize MSTs of different sizes
    # for size in [4, 8, 12]:
    #     visualize_mst(size)
    visualize_multiple_msts()