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

def worker_with_timeout(rank, size, topology, test_type, results_queue, timeout_seconds=30):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        worker(rank, size, topology, test_type, results_queue)
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

def worker(rank: int, size: int, topology: str, test_type: str, results_queue: mp.Queue):
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
        my_tensor = np.random.randint(1, 100, size=(1, 5), dtype=np.int64)
        if test_type == 'broadcast':
            try:
                send_data = my_tensor if rank == 0 else None
                received_data = comm.broadcast(send_data)
                results_queue.put({
                    'rank': rank,
                    'topology': topology,
                    'data': received_data.tolist() if isinstance(received_data, np.ndarray) else received_data,
                    'local_data': my_tensor.tolist(),
                    'error': False
                })
            except Exception as e:
                raise RuntimeError(f"Broadcast operation error: {str(e)}")
        elif test_type == 'gather':
            try:
                gathered_data = comm.gather(my_tensor)
                results_queue.put({
                    'rank': rank,
                    'topology': topology,
                    'data': gathered_data.tolist() if rank == 0 and isinstance(gathered_data, np.ndarray) else 
                            [arr.tolist() for arr in gathered_data] if rank == 0 and isinstance(gathered_data, list) else
                            my_tensor.tolist(),
                    'local_data': my_tensor.tolist(),
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
                        scatter_data.append(np.random.randint(1, 100, size=(1, 5), dtype=np.int64))
                else:
                    scatter_data = None
                received_data = comm.scatter(scatter_data)
                results_queue.put({
                    'rank': rank,
                    'topology': topology,
                    'data': received_data.tolist() if isinstance(received_data, np.ndarray) else received_data,
                    'local_data': my_tensor.tolist(),
                    'full_data': [arr.tolist() for arr in scatter_data] if rank == 0 and scatter_data is not None else None,
                    'error': False
                })
            except Exception as e:
                raise RuntimeError(f"Scatter operation error: {str(e)}")
        elif test_type == 'reduce':
            try:
                reduce_op = lambda a, b: np.add(a, b)
                reduced_data = comm.reduce(my_tensor, reduce_op)
                results_queue.put({
                    'rank': rank,
                    'topology': topology,
                    'data': reduced_data.tolist() if rank == 0 and isinstance(reduced_data, np.ndarray) else None,
                    'local_data': my_tensor.tolist(),
                    'error': False
                })
            except Exception as e:
                raise RuntimeError(f"Reduce operation error: {str(e)}")
        elif test_type == 'allreduce':
            try:
                reduce_op = lambda a, b: np.add(a, b)
                allreduced_data = comm.allreduce(my_tensor, reduce_op)
                results_queue.put({
                    'rank': rank,
                    'topology': topology,
                    'data': allreduced_data.tolist() if isinstance(allreduced_data, np.ndarray) else allreduced_data,
                    'local_data': my_tensor.tolist(),
                    'error': False
                })
            except Exception as e:
                raise RuntimeError(f"Allreduce operation error: {str(e)}")
        elif test_type == 'allgather':
            try:
                allgathered_data = comm.allgather(my_tensor)
                results_queue.put({
                    'rank': rank,
                    'topology': topology,
                    'data': [arr.tolist() for arr in allgathered_data] if isinstance(allgathered_data, list) else
                            allgathered_data.tolist() if isinstance(allgathered_data, np.ndarray) else allgathered_data,
                    'local_data': my_tensor.tolist(),
                    'error': False
                })
            except Exception as e:
                raise RuntimeError(f"Allgather operation error: {str(e)}")
        elif test_type == 'reduce_scatter':
            try:
                scatter_data = []
                for i in range(size):
                    np.random.seed(i + 1000 + test_type_seed * 100 + size * 10 + topology_seed + rank * 1000)
                    scatter_data.append(np.random.randint(1, 100, size=(1, 5), dtype=np.int64))
                reduce_op = lambda a, b: np.add(a, b)
                reduced_scattered_data = comm.reduce_scatter(scatter_data, reduce_op)
                results_queue.put({
                    'rank': rank,
                    'topology': topology,
                    'data': reduced_scattered_data.tolist() if isinstance(reduced_scattered_data, np.ndarray) else reduced_scattered_data,
                    'local_data': [arr.tolist() for arr in scatter_data],
                    'error': False
                })
            except Exception as e:
                raise RuntimeError(f"Reduce_scatter operation error: {str(e)}")
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

def run_test(test_type: str, topology_name: str, size: int):
    log(f"\n\n=== Testing {test_type} on {topology_name} topology with {size} nodes ===")
    results = mp.Queue()
    processes = []
    for rank in range(size):
        p = mp.Process(
            target=worker_with_timeout,
            args=(rank, size, topology_name, test_type, results)
        )
        processes.append(p)
        p.start()
        log(f"{Colors.BLUE}Started process for rank {rank}{Colors.RESET}")
    for i, p in enumerate(processes):
        p.join(timeout=45)
        if p.is_alive():
            log(f"{Colors.RED}Warning: Process for rank {i} did not terminate, forcing termination{Colors.RESET}")
            p.terminate()
            p.join(1)
            if p.is_alive():
                log(f"{Colors.RED}Error: Unable to terminate process for rank {i}{Colors.RESET}")
                try:
                    import os
                    import signal
                    os.kill(p.pid, signal.SIGKILL)
                except:
                    pass
    all_results = []
    try:
        start_time = time.time()
        timeout = 5
        expected_results = size
        while len(all_results) < expected_results and (time.time() - start_time) < timeout:
            try:
                if not results.empty():
                    all_results.append(results.get(block=False))
                else:
                    time.sleep(0.1)
            except Exception as e:
                log(f"{Colors.YELLOW}Warning while collecting results: {str(e)}{Colors.RESET}")
        if len(all_results) < expected_results:
            log(f"{Colors.RED}Warning: Expected {expected_results} results, but received only {len(all_results)}{Colors.RESET}")
    except Exception as e:
        log(f"{Colors.RED}Error while collecting results: {str(e)}{Colors.RESET}")
    try:
        all_results.sort(key=lambda x: x['rank'])
    except Exception as e:
        log(f"{Colors.RED}Error while sorting results: {str(e)}{Colors.RESET}")
    received_ranks = set(res.get('rank', -1) for res in all_results)
    missing_ranks = set(range(size)) - received_ranks
    if missing_ranks:
        log(f"{Colors.RED}Missing results from the following ranks: {missing_ranks}{Colors.RESET}")
        for rank in missing_ranks:
            all_results.append({
                'rank': rank,
                'topology': topology_name,
                'data': None,
                'error': True,
                'error_message': "No result received from this rank"
            })
        all_results.sort(key=lambda x: x['rank'])
    any_errors = any(res.get('error', False) for res in all_results)
    if any_errors:
        log(f"\n{Colors.RED}Errors occurred during test execution!{Colors.RESET}")
        for res in all_results:
            if res.get('error', False):
                log(f"{Colors.RED}Node {res['rank']} error: {res.get('error_message', 'Unknown error')}{Colors.RESET}")
                if 'error_traceback' in res:
                    log(f"{Colors.RED}Stack trace: {res['error_traceback']}{Colors.RESET}")
        log(f"\n{Colors.RED}Test failed due to execution errors{Colors.RESET}")
        return False, test_type, topology_name, size
    node_data = {}
    for res in all_results:
        if 'local_data' in res:
            node_data[res['rank']] = res['local_data']
    test_passed = False
    if test_type == 'broadcast':
        broadcast_correct = validate_broadcast_results(all_results, size, node_data)
        status = f"{Colors.GREEN}✓ Correct{Colors.RESET}" if broadcast_correct else f"{Colors.RED}✗ Incorrect{Colors.RESET}"
        log(f"  Validation result: {status}")
        test_passed = broadcast_correct
    elif test_type == 'gather':
        gather_correct = validate_gather_results(all_results, size, node_data)
        status = f"{Colors.GREEN}✓ Correct{Colors.RESET}" if gather_correct else f"{Colors.RED}✗ Incorrect{Colors.RESET}"
        log(f"  Validation result: {status}")
        test_passed = gather_correct
    elif test_type == 'scatter':
        scatter_correct = validate_scatter_results(all_results, size, node_data)
        status = f"{Colors.GREEN}✓ Correct{Colors.RESET}" if scatter_correct else f"{Colors.RED}✗ Incorrect{Colors.RESET}"
        log(f"  Validation result: {status}")
        test_passed = scatter_correct
    elif test_type == 'reduce':
        reduce_correct = validate_reduce_results(all_results, size, node_data)
        status = f"{Colors.GREEN}✓ Correct{Colors.RESET}" if reduce_correct else f"{Colors.RED}✗ Incorrect{Colors.RESET}"
        log(f"  Validation result: {status}")
        test_passed = reduce_correct
    elif test_type == 'allreduce':
        allreduce_correct = validate_allreduce_results(all_results, size, node_data)
        status = f"{Colors.GREEN}✓ Correct{Colors.RESET}" if allreduce_correct else f"{Colors.RED}✗ Incorrect{Colors.RESET}"
        log(f"  Validation result: {status}")
        test_passed = allreduce_correct
    elif test_type == 'allgather':
        allgather_correct = validate_allgather_results(all_results, size, node_data)
        status = f"{Colors.GREEN}✓ Correct{Colors.RESET}" if allgather_correct else f"{Colors.RED}✗ Incorrect{Colors.RESET}"
        log(f"  Validation result: {status}")
        test_passed = allgather_correct
    elif test_type == 'reduce_scatter':
        reduce_scatter_correct = validate_reduce_scatter_results(all_results, size, node_data)
        status = f"{Colors.GREEN}✓ Correct{Colors.RESET}" if reduce_scatter_correct else f"{Colors.RED}✗ Incorrect{Colors.RESET}"
        log(f"  Validation result: {status}")
        test_passed = reduce_scatter_correct
    log(f"\nOverall result: {Colors.GREEN+'✓ All correct'+Colors.RESET if test_passed else Colors.RED+'✗ Errors found'+Colors.RESET}")
    print_topology(topology_name, size)
    return test_passed, test_type, topology_name, size

def print_topology(topology_name: str, size: int):
    log("\nTopology structure:")
    topo_obj = None
    if topology_name == 'ring':
        from topology.ring import RingTopology
        topo_obj = RingTopology()
    elif topology_name == 'tree':
        from topology.tree import TreeTopology
        topo_obj = TreeTopology()
    elif topology_name == 'star':
        from topology.star import StarTopology
        topo_obj = StarTopology()
    if topo_obj:
        for rank in range(size):
            children = topo_obj.get_children(rank, size)
            parent = topo_obj.get_parent(rank, size)
            log(f"  Node {rank}: Parent={parent}, Children={children}")

def run_all_tests(test_type: str, topologies: list, sizes: list):
    log(f"\n========== {test_type} Tests ==========")
    test_results = []
    all_passed = True
    for size in sizes:
        for topology in topologies:
            test_passed, prim, topo, sz = run_test(test_type, topology, size)
            all_passed = all_passed and test_passed
            test_results.append({
                'primitive': prim,
                'topology': topo, 
                'size': sz,
                'passed': test_passed
            })
    return all_passed, test_results

def print_summary(all_results):
    log("\n\n=========================================")
    log("=           Test Result Summary         =")
    log("=========================================")
    primitives = sorted(set(r['primitive'] for r in all_results))
    topologies = sorted(set(r['topology'] for r in all_results))
    sizes = sorted(set(r['size'] for r in all_results))
    headers = ["Operation"] + [f"{t} topology\n({s} nodes)" for t in topologies for s in sizes]
    table_data = []
    for prim in primitives:
        row = [prim]
        for topo in topologies:
            for size in sizes:
                found = False
                for r in all_results:
                    if r['primitive'] == prim and r['topology'] == topo and r['size'] == size:
                        status = f"{Colors.GREEN}✓ Pass{Colors.RESET}" if r['passed'] else f"{Colors.RED}✗ Fail{Colors.RESET}"
                        row.append(status)
                        found = True
                        break
                if not found:
                    row.append("Not tested")
        table_data.append(row)
    passed_count = sum(1 for r in all_results if r['passed'])
    total_count = len(all_results)
    pass_rate = 100.0 * passed_count / total_count if total_count > 0 else 0.0
    log(tabulate(table_data, headers=headers, tablefmt="grid"))
    log(f"\nTotal tests: {total_count}")
    log(f"Passed tests: {passed_count}")
    log(f"Failed tests: {total_count - passed_count}")
    log(f"Pass rate: {pass_rate:.1f}%")
    if total_count - passed_count > 0:
        log("\nFailed tests:")
        for r in all_results:
            if not r['passed']:
                log(f"  • {r['primitive']} - {r['topology']} topology ({r['size']} nodes)")

def validate_broadcast_results(all_results, size, node_data):
    broadcast_correct = True
    log("\n== Broadcast Test Results ==")
    log("\nOriginal data for each node:")
    for rank in range(size):
        if rank in node_data:
            log(f"  Node {rank} original data: {node_data[rank]}")
    root_data = node_data.get(0)
    if root_data is None:
        log(f"{Colors.RED}Missing root node data{Colors.RESET}")
        return False
    for res in all_results:
        is_correct = res['data'] == root_data
        broadcast_correct = broadcast_correct and is_correct
        status = f"{Colors.GREEN}✓ Correct{Colors.RESET}" if is_correct else f"{Colors.RED}✗ Incorrect{Colors.RESET}"
        log(f"  Node {res['rank']}: Received {res['data']} - {status}")
        if not is_correct:
            log(f"    Expected: {root_data}")
    return broadcast_correct

def validate_gather_results(all_results, size, node_data):
    gather_correct = True
    log("\n== Gather Test Results ==")
    log("\nOriginal data for each node:")
    for rank in range(size):
        if rank in node_data:
            log(f"  Node {rank} original data: {node_data[rank]}")
    root_result = None
    for res in all_results:
        if res['rank'] == 0:
            root_result = res['data']
            break
    if root_result is None:
        log(f"{Colors.RED}Missing root node result{Colors.RESET}")
        return False
    expected = []
    for i in range(size):
        if i in node_data:
            expected.append(node_data[i])
    is_correct = root_result == expected
    gather_correct = is_correct
    status = f"{Colors.GREEN}✓ Correct{Colors.RESET}" if is_correct else f"{Colors.RED}✗ Incorrect{Colors.RESET}"
    log(f"  Node 0 (root): Gathered {root_result} - {status}")
    if not is_correct:
        log(f"    Expected: {expected}")
    return gather_correct

def validate_scatter_results(all_results, size, node_data):
    scatter_correct = True
    log("\n== Scatter Test Results ==")
    log("\nOriginal data for each node:")
    for rank in range(size):
        if rank in node_data:
            log(f"  Node {rank} original data: {node_data[rank]}")
    root_full_data = None
    for res in all_results:
        if res['rank'] == 0 and 'full_data' in res:
            root_full_data = res['full_data']
            break
    if root_full_data is None:
        log(f"{Colors.RED}Missing root node full data{Colors.RESET}")
        return False
    log(f"\nFull data list sent by root node: {root_full_data}")
    for res in all_results:
        expected = root_full_data[res['rank']]
        is_correct = res['data'] == expected
        scatter_correct = scatter_correct and is_correct
        status = f"{Colors.GREEN}✓ Correct{Colors.RESET}" if is_correct else f"{Colors.RED}✗ Incorrect{Colors.RESET}"
        log(f"  Node {res['rank']}: Received {res['data']} - {status}")
        if not is_correct:
            log(f"    Expected: {expected}")
    return scatter_correct

def validate_reduce_results(all_results, size, node_data):
    reduce_correct = True
    log("\n== Reduce Test Results ==")
    log("\nOriginal data for each node:")
    for rank in range(size):
        if rank in node_data:
            log(f"  Node {rank} original data: {node_data[rank]}")
    expected_sum = np.zeros((1, 5), dtype=np.int64)
    for rank, data in node_data.items():
        expected_sum += np.array(data, dtype=np.int64)
    log(f"\nExpected reduction result (sum of all node data): {expected_sum.tolist()}")
    for res in all_results:
        if res['rank'] == 0:
            is_correct = res['data'] == expected_sum.tolist()
            reduce_correct = is_correct
            status = f"{Colors.GREEN}✓ Correct{Colors.RESET}" if is_correct else f"{Colors.RED}✗ Incorrect{Colors.RESET}"
            log(f"  Node 0 (root): Reduction result {res['data']} - {status}")
            if not is_correct:
                log(f"    Expected: {expected_sum.tolist()}")
            break
    return reduce_correct

def validate_allreduce_results(all_results, size, node_data):
    allreduce_correct = True
    log("\n== Allreduce Test Results ==")
    log("\nOriginal data for each node:")
    for rank in range(size):
        if rank in node_data:
            log(f"  Node {rank} original data: {node_data[rank]}")
    expected_sum = np.zeros((1, 5), dtype=np.int64)
    for rank, data in node_data.items():
        expected_sum += np.array(data, dtype=np.int64)
    log(f"\nExpected reduction result (sum of all node data): {expected_sum.tolist()}")
    for res in all_results:
        is_correct = res['data'] == expected_sum.tolist()
        allreduce_correct = allreduce_correct and is_correct
        status = f"{Colors.GREEN}✓ Correct{Colors.RESET}" if is_correct else f"{Colors.RED}✗ Incorrect{Colors.RESET}"
        log(f"  Node {res['rank']}: Received {res['data']} - {status}")
        if not is_correct:
            log(f"    Expected: {expected_sum.tolist()}")
    return allreduce_correct

def validate_allgather_results(all_results, size, node_data):
    allgather_correct = True
    log("\n== Allgather Test Results ==")
    log("\nOriginal data for each node:")
    for rank in range(size):
        if rank in node_data:
            log(f"  Node {rank} original data: {node_data[rank]}")
    expected = []
    for i in range(size):
        if i in node_data:
            expected.append(node_data[i])
    log(f"\nExpected gathered result (list of all node data): {expected}")
    for res in all_results:
        is_correct = res['data'] == expected
        allgather_correct = allgather_correct and is_correct
        status = f"{Colors.GREEN}✓ Correct{Colors.RESET}" if is_correct else f"{Colors.RED}✗ Incorrect{Colors.RESET}"
        log(f"  Node {res['rank']}: Received {res['data']} - {status}")
        if not is_correct:
            log(f"    Expected: {expected}")
    return allgather_correct

def validate_reduce_scatter_results(all_results, size, node_data):
    reduce_scatter_correct = True
    log("\n== Reduce Scatter Test Results ==")
    log("\nInput data for each node:")
    for res in all_results:
        if 'local_data' in res and isinstance(res['local_data'], list):
            log(f"  Node {res['rank']} input data list: {res['local_data']}")
    combined_inputs = []
    for i in range(size):
        position_data = []
        for res in all_results:
            if 'local_data' in res and isinstance(res['local_data'], list) and len(res['local_data']) > i:
                position_data.append(np.array(res['local_data'][i]))
        combined_inputs.append(position_data)
    expected_results = []
    for i in range(size):
        if combined_inputs[i]:
            expected_sum = np.zeros_like(combined_inputs[i][0])
            for arr in combined_inputs[i]:
                expected_sum = np.add(expected_sum, arr)
            expected_results.append(expected_sum.tolist())
    log(f"\nExpected reduction results: {expected_results}")
    for res in all_results:
        rank = res['rank']
        if rank < len(expected_results):
            expected = expected_results[rank]
            is_correct = res['data'] == expected
            reduce_scatter_correct = reduce_scatter_correct and is_correct
            status = f"{Colors.GREEN}✓ Correct{Colors.RESET}" if is_correct else f"{Colors.RED}✗ Incorrect{Colors.RESET}"
            log(f"  Node {rank}: Received {res['data']} - {status}")
            if not is_correct:
                log(f"    Expected: {expected}")
    return reduce_scatter_correct

if __name__ == '__main__':
    log(f"Test started at: {timestamp}")
    log(f"Log file: {log_filename}")
    sizes = [4, 8, 16]
    topologies = ['ring', 'tree']    
    primitives = [
        'broadcast',
        'gather',
        'scatter',
        'reduce',
        'allreduce',
        'allgather',
        'reduce_scatter'
    ]
    all_results = []
    all_passed = True
    for primitive in primitives:
        primitive_passed, test_results = run_all_tests(primitive, topologies, sizes)
        all_passed = all_passed and primitive_passed
        all_results.extend(test_results)
    print_summary(all_results)
    log("\n=== Test Summary ===")
    log(f"Test result: {Colors.GREEN+'All tests passed'+Colors.RESET if all_passed else Colors.RED+'Some tests failed'+Colors.RESET}")
    log(f"\nTest ended at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Total time: {(datetime.datetime.now() - datetime.datetime.strptime(timestamp, '%Y%m%d_%H%M%S')).total_seconds():.2f} seconds")
    log_file.close()
    print(f"\nTest log saved to: {log_filename}")
