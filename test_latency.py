from communicator import Communicator
import multiprocessing as mp
import time
import signal
import traceback
from tabulate import tabulate
import datetime
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = os.path.join(log_dir, f"collective_test_rand_log_{timestamp}.txt")
log_file = open(log_filename, "w", encoding="utf-8")

def log(message):
    print(message)
    clean_message = message.replace(Colors.GREEN, "").replace(Colors.RED, "").replace(Colors.YELLOW, "").replace(Colors.BLUE, "").replace(Colors.RESET, "")
    log_file.write(clean_message + "\n")
    log_file.flush()

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Test operation timed out")

def get_tensor_size_for_data_volume(data_volume_kb):
    bytes_per_element = 8
    elements = (data_volume_kb * 1024) // bytes_per_element
    return elements

def worker_with_timeout(rank, size, topology, test_type, results_queue, timeout_seconds=30, data_volume_kb=1):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        worker(rank, size, topology, test_type, results_queue, data_volume_kb)
    except TimeoutError as e:
        results_queue.put({
            'rank': rank,
            'topology': topology,
            'data': None,
            'error': True,
            'error_message': f"Timeout: Operation took longer than {timeout_seconds} seconds",
            'error_traceback': traceback.format_exc()
        })
    finally:
        signal.alarm(0)

def worker(rank: int, size: int, topology: str, test_type: str, results_queue: mp.Queue, data_volume_kb: int = 1):
    comm = None
    try:
        try:
            comm = Communicator(rank, size, topology)
            log(f"Node {rank}: Communicator initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Communicator initialization failed: {str(e)}")
        time.sleep(0.5)
        test_type_seed = sum(ord(c) for c in test_type)
        topology_seed = sum(ord(c) for c in topology)
        seed_value = rank + 1000 + test_type_seed * 100 + size * 10 + topology_seed
        np.random.seed(seed_value)
        tensor_size = get_tensor_size_for_data_volume(data_volume_kb)
        my_tensor = np.random.randint(1, 100, size=(1, 128), dtype=np.int64)
        print(f"Node {rank}: tensor_size: {tensor_size}, size: {size}")
        start_time = time.time()
        if test_type == 'broadcast':
            try:
                send_data = np.random.randint(1, 100, size=(1, int(tensor_size * size)), dtype=np.int64) if rank == 0 else None
                received_data = comm.broadcast(send_data)
                end_time = time.time()
                results_queue.put({
                    'rank': rank,
                    'topology': topology,
                    'data': received_data.tolist() if isinstance(received_data, np.ndarray) else received_data,
                    'local_data': send_data.tolist() if rank == 0 and isinstance(send_data, np.ndarray) else None,
                    'start_time': start_time,
                    'end_time': end_time,
                    'error': False
                })
            except Exception as e:
                raise RuntimeError(f"Broadcast operation error: {str(e)}")
        elif test_type == 'gather':
            try:
                my_tensor = np.random.randint(1, 100, size=(1, int(tensor_size)), dtype=np.int64)
                gathered_data = comm.gather(my_tensor)
                end_time = time.time()
                results_queue.put({
                    'rank': rank,
                    'topology': topology,
                    'data': gathered_data.tolist() if rank == 0 and isinstance(gathered_data, np.ndarray) else 
                            [arr.tolist() for arr in gathered_data] if rank == 0 and isinstance(gathered_data, list) else
                            my_tensor.tolist(),
                    'local_data': my_tensor.tolist(),
                    'start_time': start_time,
                    'end_time': end_time,
                    'error': False
                })
            except Exception as e:
                raise RuntimeError(f"Gather operation error: {str(e)}")
        elif test_type == 'scatter':
            try:
                if rank == 0:
                    scatter_data = []
                    for i in range(size):
                        np.random.seed(i + 1000 + test_type_seed * 100 + size * 10 + topology_seed)
                        scatter_data.append(np.random.randint(1, 100, size=(1, int(tensor_size)), dtype=np.int64))
                else:
                    scatter_data = None
                received_data = comm.scatter(scatter_data)
                end_time = time.time()
                results_queue.put({
                    'rank': rank,
                    'topology': topology,
                    'data': received_data.tolist() if isinstance(received_data, np.ndarray) else received_data,
                    'local_data': received_data.tolist() if isinstance(received_data, np.ndarray) else None,
                    'full_data': [arr.tolist() for arr in scatter_data] if rank == 0 and scatter_data is not None else None,
                    'start_time': start_time,
                    'end_time': end_time,
                    'error': False
                })
            except Exception as e:
                raise RuntimeError(f"Scatter operation error: {str(e)}")
        elif test_type == 'reduce':
            try:
                my_tensor = np.random.randint(1, 100, size=(1, int(tensor_size * size)), dtype=np.int64)
                reduce_op = lambda a, b: np.add(a, b)
                reduced_data = comm.reduce(my_tensor, reduce_op)
                end_time = time.time()
                results_queue.put({
                    'rank': rank,
                    'topology': topology,
                    'data': reduced_data.tolist() if rank == 0 and isinstance(reduced_data, np.ndarray) else None,
                    'local_data': my_tensor.tolist(),
                    'start_time': start_time,
                    'end_time': end_time,
                    'error': False
                })
            except Exception as e:
                raise RuntimeError(f"Reduce operation error: {str(e)}")
        elif test_type == 'allreduce':
            try:
                my_tensor = np.random.randint(1, 100, size=(1, int(tensor_size * size)), dtype=np.int64)
                reduce_op = lambda a, b: np.add(a, b)
                allreduced_data = comm.allreduce(my_tensor, reduce_op)
                end_time = time.time()
                results_queue.put({
                    'rank': rank,
                    'topology': topology,
                    'data': allreduced_data.tolist() if isinstance(allreduced_data, np.ndarray) else allreduced_data,
                    'local_data': my_tensor.tolist(),
                    'start_time': start_time,
                    'end_time': end_time,
                    'error': False
                })
            except Exception as e:
                raise RuntimeError(f"Allreduce operation error: {str(e)}")
        elif test_type == 'allgather':
            try:
                my_tensor = np.random.randint(1, 100, size=(1, int(tensor_size)), dtype=np.int64)
                allgathered_data = comm.allgather(my_tensor)
                end_time = time.time()
                results_queue.put({
                    'rank': rank,
                    'topology': topology,
                    'data': [arr.tolist() for arr in allgathered_data] if isinstance(allgathered_data, list) else
                            allgathered_data.tolist() if isinstance(allgathered_data, np.ndarray) else allgathered_data,
                    'local_data': my_tensor.tolist(),
                    'start_time': start_time,
                    'end_time': end_time,
                    'error': False
                })
            except Exception as e:
                raise RuntimeError(f"Allgather operation error: {str(e)}")
        elif test_type == 'reduce_scatter':
            try:
                scatter_data = []
                for i in range(size):
                    np.random.seed(i + 1000 + test_type_seed * 100 + size * 10 + topology_seed + rank * 1000)
                    scatter_data.append(np.random.randint(1, 100, size=(1, int(tensor_size)), dtype=np.int64))
                reduce_op = lambda a, b: np.add(a, b)
                reduced_scattered_data = comm.reduce_scatter(scatter_data, reduce_op)
                end_time = time.time()
                results_queue.put({
                    'rank': rank,
                    'topology': topology,
                    'data': reduced_scattered_data.tolist() if isinstance(reduced_scattered_data, np.ndarray) else reduced_scattered_data,
                    'local_data': [arr.tolist() for arr in scatter_data],
                    'start_time': start_time,
                    'end_time': end_time,
                    'error': False
                })
            except Exception as e:
                raise RuntimeError(f"Reduce scatter operation error: {str(e)}")
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    except Exception as e:
        error_message = f"{test_type} test failed on node {rank}: {str(e)}"
        error_traceback = traceback.format_exc()
        log(f"\n{Colors.RED}Worker process exception (rank {rank}):{Colors.RESET}")
        log(f"{Colors.RED}{error_message}{Colors.RESET}")
        log(f"{Colors.RED}{error_traceback}{Colors.RESET}")
        results_queue.put({
            'rank': rank,
            'topology': topology,
            'data': None,
            'error': True,
            'error_message': error_message,
            'error_traceback': error_traceback
        })
    finally:
        if comm:
            try:
                comm.close()
            except Exception as e:
                log(f"{Colors.YELLOW}Warning: Error closing communicator for rank {rank}: {str(e)}{Colors.RESET}")

def run_latency_test(test_type: str, topology_name: str, size: int, data_volume_kb: int = 1, iterations: int = 5):
    log(f"\n\n=== Latency Test: {test_type} on {topology_name} topology, {size} nodes, {iterations} iterations, Data volume: {data_volume_kb}KB ===")
    latencies = []
    for i in range(iterations):
        log(f"\nIteration {i+1}/{iterations}")
        results = mp.Queue()
        processes = []
        for rank in range(size):
            p = mp.Process(
                target=worker_with_timeout,
                args=(rank, size, topology_name, test_type, results, 45, data_volume_kb)
            )
            processes.append(p)
            p.start()
            
        # 先收集队列结果
        all_results = []
        start_time = time.time()
        timeout = 5
        while len(all_results) < size and (time.time() - start_time) < timeout:
            try:
                if not results.empty():
                    all_results.append(results.get(block=False))
                else:
                    time.sleep(0.1)
            except:
                pass
                
        # 再join进程
        for p in processes:
            p.join(timeout=45)
            if p.is_alive():
                p.terminate()
                p.join(1)
                
        if len(all_results) < size:
            log(f"{Colors.YELLOW}Warning: Expected {size} results, but received {len(all_results)}{Colors.RESET}")
            continue
        try:
            start_time = min(res.get('start_time', float('inf')) for res in all_results if not res.get('error', False))
            end_time = max(res.get('end_time', 0) for res in all_results if not res.get('error', False))
            iteration_latency = (end_time - start_time) * 1000
            latencies.append(iteration_latency)
            log(f"Iteration {i+1} latency: {iteration_latency:.3f} ms")
        except Exception as e:
            log(f"{Colors.RED}Error calculating latency: {str(e)}{Colors.RESET}")
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        if len(latencies) > 1:
            std_dev = (sum((x - avg_latency) ** 2 for x in latencies) / len(latencies)) ** 0.5
        else:
            std_dev = 0
        log(f"\n{Colors.GREEN}=== Latency Statistics ==={Colors.RESET}")
        log(f"Operation: {test_type}")
        log(f"Topology: {topology_name}")
        log(f"Number of nodes: {size}")
        log(f"Number of iterations: {len(latencies)}")
        log(f"Average latency: {avg_latency:.3f} ms")
        log(f"Standard deviation: {std_dev:.3f} ms")
        log(f"Minimum: {min(latencies):.3f} ms")
        log(f"Maximum: {max(latencies):.3f} ms")
        return {
            'primitive': test_type,
            'topology': topology_name,
            'size': size,
            'data_volume_kb': data_volume_kb,
            'avg_latency': avg_latency,
            'std_dev': std_dev,
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'iterations': len(latencies)
        }
    return None

def plot_performance_comparison(results, plot_dir):
    primitives = sorted(set(r['primitive'] for r in results))
    topologies = sorted(set(r['topology'] for r in results))
    node_sizes = sorted(set(r['size'] for r in results))
    data_sizes = sorted(set(r['data_volume_kb'] for r in results))
    colors = [
        '#4e79a7', '#f28e2c', '#e15759', '#76b7b2', 
        '#59a14f', '#edc949', '#af7aa1', '#ff9da7',
        '#9c755f', '#bab0ab'
    ]
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8), facecolor='#f9f9f9')
    n_groups = len(topologies) * len(node_sizes)
    index = np.arange(len(primitives))
    bar_width = 0.8 / n_groups
    for i, topology in enumerate(topologies):
        for j, nodes in enumerate(node_sizes):
            max_data_size = max(data_sizes) if data_sizes else 1
            values = []
            for primitive in primitives:
                data_points = [r for r in results if r['topology'] == topology 
                              and r['primitive'] == primitive 
                              and r['size'] == nodes
                              and r['data_volume_kb'] == max_data_size]
                if data_points:
                    values.append(data_points[0]['avg_latency'])
                else:
                    values.append(0)
            position = index + bar_width * (i * len(node_sizes) + j)
            color_idx = (i * len(node_sizes) + j) % len(colors)
            plt.bar(position, values, bar_width,
                   color=colors[color_idx], 
                   edgecolor='white', linewidth=0.7,
                   label=f"{topology}-{nodes}nodes")
    plt.xlabel('Communication Primitives', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.title(f'Latency Comparison for Different Topologies and Node Counts ({max_data_size}KB data)', fontsize=14, pad=20)
    plt.xticks(index + bar_width * (n_groups/2 - 0.5), primitives, fontsize=10)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, facecolor='white', edgecolor='#e0e0e0')
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='#e0e0e0')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "topology_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(14, 8), facecolor='#f9f9f9')
    main_topology = topologies[0] if topologies else "ring"
    main_nodes = max(node_sizes) if node_sizes else 4
    index = np.arange(len(data_sizes))
    bar_width = 0.8 / len(primitives)
    for i, primitive in enumerate(primitives):
        values = []
        for data_size in data_sizes:
            data_points = [r for r in results if r['topology'] == main_topology 
                          and r['primitive'] == primitive 
                          and r['size'] == main_nodes
                          and r['data_volume_kb'] == data_size]
            if data_points:
                values.append(data_points[0]['avg_latency'])
            else:
                values.append(0)
        position = index + i * bar_width
        color_idx = i % len(colors)
        plt.bar(position, values, bar_width,
               color=colors[color_idx],
               edgecolor='white', linewidth=0.7,
               label=primitive)
    plt.xlabel('Data Size (KB)', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.title(f'Impact of Data Size on {main_topology} Topology Latency ({main_nodes} nodes)', fontsize=14, pad=20)
    plt.xticks(index + bar_width * (len(primitives)/2 - 0.5), [str(size) for size in data_sizes], fontsize=10)
    plt.legend(frameon=True, facecolor='white', edgecolor='#e0e0e0', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='#e0e0e0')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "data_size_impact.png"), dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(14, 8), facecolor='#f9f9f9')
    max_data_size = max(data_sizes) if data_sizes else 1
    main_topology = topologies[0] if topologies else "ring"
    index = np.arange(len(node_sizes))
    bar_width = 0.8 / len(primitives)
    for i, primitive in enumerate(primitives):
        values = []
        for nodes in node_sizes:
            data_points = [r for r in results if r['topology'] == main_topology 
                          and r['primitive'] == primitive 
                          and r['size'] == nodes
                          and r['data_volume_kb'] == max_data_size]
            if data_points:
                values.append(data_points[0]['avg_latency'])
            else:
                values.append(0)
        position = index + i * bar_width
        color_idx = i % len(colors)
        plt.bar(position, values, bar_width,
               color=colors[color_idx],
               edgecolor='white', linewidth=0.7,
               label=primitive)
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.title(f'Impact of Node Count on {main_topology} Topology Latency ({max_data_size}KB data)', fontsize=14, pad=20)
    plt.xticks(index + bar_width * (len(primitives)/2 - 0.5), [str(size) for size in node_sizes], fontsize=10)
    plt.legend(frameon=True, facecolor='white', edgecolor='#e0e0e0', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='#e0e0e0')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "node_count_impact.png"), dpi=300, bbox_inches='tight')
    plt.close()
    combo_labels = []
    for topology in topologies:
        for nodes in node_sizes:
            for data_size in data_sizes:
                combo_labels.append(f"{topology}-{nodes}n-{data_size}KB")
    heatmap_data = np.zeros((len(combo_labels), len(primitives)))
    for i, combo in enumerate(combo_labels):
        topology, node_size, data_size = combo.split('-')
        node_size = int(node_size.replace('n', ''))
        data_size = float(data_size.replace('KB', ''))
        for j, primitive in enumerate(primitives):
            data_points = [r for r in results if r['topology'] == topology 
                         and r['primitive'] == primitive 
                         and r['size'] == node_size
                         and r['data_volume_kb'] == data_size]
            if data_points:
                heatmap_data[i, j] = data_points[0]['avg_latency']
    plt.figure(figsize=(16, max(10, len(combo_labels)/2)), facecolor='#f9f9f9')
    cmap = plt.cm.get_cmap('YlOrRd')
    im = plt.pcolormesh(heatmap_data, cmap=cmap)
    cbar = plt.colorbar(im)
    cbar.set_label('Latency (ms)', fontsize=12)
    plt.xticks(np.arange(len(primitives)) + 0.5, primitives, fontsize=10, rotation=45, ha='right')
    plt.yticks(np.arange(len(combo_labels)) + 0.5, combo_labels, fontsize=10)
    plt.title('Global Performance Heatmap (Higher latency is redder)', fontsize=14, pad=20)
    plt.tight_layout()
    for i in range(len(combo_labels)):
        for j in range(len(primitives)):
            if heatmap_data[i, j] > 0:
                text_color = 'white' if heatmap_data[i, j] > np.max(heatmap_data)/2 else 'black'
                plt.text(j + 0.5, i + 0.5, f"{heatmap_data[i, j]:.1f}", 
                       ha="center", va="center", fontsize=9,
                       color=text_color)
    plt.savefig(os.path.join(plot_dir, "global_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    for data_size in data_sizes:
        heatmap_matrix = np.zeros((len(node_sizes), len(primitives) * len(topologies)))
        col_labels = []
        for primitive in primitives:
            for topology in topologies:
                col_labels.append(f"{primitive}-{topology}")
        for i, nodes in enumerate(node_sizes):
            for j, (primitive, topology) in enumerate([(p, t) for p in primitives for t in topologies]):
                data_points = [r for r in results if r['topology'] == topology 
                             and r['primitive'] == primitive 
                             and r['size'] == nodes
                             and r['data_volume_kb'] == data_size]
                if data_points:
                    heatmap_matrix[i, j] = data_points[0]['avg_latency']
        plt.figure(figsize=(max(12, len(col_labels)*0.8), len(node_sizes)*0.8), facecolor='#f9f9f9')
        im = plt.pcolormesh(heatmap_matrix, cmap=cmap)
        cbar = plt.colorbar(im)
        cbar.set_label('Latency (ms)', fontsize=12)
        plt.xticks(np.arange(len(col_labels)) + 0.5, col_labels, fontsize=9, rotation=90)
        plt.yticks(np.arange(len(node_sizes)) + 0.5, [f"{n}nodes" for n in node_sizes], fontsize=10)
        plt.title(f'Performance Heatmap ({data_size}KB data)', fontsize=14, pad=20)
        plt.tight_layout()
        for i in range(len(node_sizes)):
            for j in range(len(col_labels)):
                if heatmap_matrix[i, j] > 0:
                    text_color = 'white' if heatmap_matrix[i, j] > np.max(heatmap_matrix)/2 else 'black'
                    plt.text(j + 0.5, i + 0.5, f"{heatmap_matrix[i, j]:.1f}", 
                           ha="center", va="center", fontsize=8,
                           color=text_color)
        plt.savefig(os.path.join(plot_dir, f"heatmap_{data_size}KB.png"), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    log("\n\n=========================================")
    log("=           Latency Test                =")
    log("=========================================")
    data_size = [0.5, 1, 2]
    nodes_num = [4, 6, 8]
    latency_topologies = ['ring', 'tree']
    latency_primitives = [
        'broadcast',
        'gather', 
        'scatter', 
        'reduce', 
        'allreduce', 
        'allgather', 
        'reduce_scatter'
    ]
    iterations = 1
    latency_results = []
    for primitive in latency_primitives:
        for topology in latency_topologies:
            for node in nodes_num:
                for data in data_size:
                    result = run_latency_test(primitive, topology, node, data, iterations)
                    if result:
                        latency_results.append(result)
    if latency_results:
        log("\n\n=========================================")
        log(f"=     Latency Test Results     =")
        log("=========================================")
        headers = ["Operation", "Topology", "Node Count", "Data Size(KB)", "Avg Latency(ms)", "Std Dev", "Min", "Max"]
        table_data = []
        for res in latency_results:
            row = [
                res['primitive'],
                res['topology'],
                res['size'],
                res['data_volume_kb'],
                f"{res['avg_latency']:.3f}",
                f"{res['std_dev']:.3f}",
                f"{res['min_latency']:.3f}",
                f"{res['max_latency']:.3f}"
            ]
            table_data.append(row)
        table_data.sort(key=lambda x: (x[0], x[1], int(x[2])))
        log(tabulate(table_data, headers=headers, tablefmt="grid"))
        plot_dir = os.path.join(log_dir, f"plots_{timestamp}")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_performance_comparison(latency_results, plot_dir)
        log(f"\nPlots saved to: {plot_dir}")
    log(f"\nTest end time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Total duration: {(datetime.datetime.now() - datetime.datetime.strptime(timestamp, '%Y%m%d_%H%M%S')).total_seconds():.2f} seconds")
    log_file.close()
    print(f"\nTest log saved to: {log_filename}")
