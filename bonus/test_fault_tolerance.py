import sys
import os
import time
import signal
import random
import multiprocessing as mp
import logging
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tabulate import tabulate
import datetime

# Add parent directory to path for importing original modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bonus.communicator import FaultTolerantCommunicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bonus/fault_tolerance_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FaultToleranceTest")

def worker(rank, size, topology, operation, kill_ranks, delay_before_kill, use_numpy=True):
    """Worker process function"""
    try:
        # Set random seed for reproducibility
        random.seed(42 + rank)
        np.random.seed(42 + rank)
        
        # Use fault-tolerant communicator
        comm = FaultTolerantCommunicator(rank, size, topology)
        
        print(f"Node {rank}: Communicator initialized successfully")
        
        # Check if this node is marked for termination
        will_be_killed = rank in kill_ranks
        
        # Test different operations
        if operation == "broadcast":
            # Create test data
            if rank == 0:  # Broadcast root node
                if use_numpy:
                    data = np.random.rand(10, 10)  # Random matrix
                else:
                    data = [random.random() for _ in range(100)]  # Random list
                print(f"Node {rank}: Broadcasting data")
            else:
                data = None
            
            # If node will be killed, set up delayed termination
            if will_be_killed:
                kill_timer = mp.Process(target=lambda: (time.sleep(delay_before_kill), os._exit(0)))
                kill_timer.daemon = True
                kill_timer.start()
                print(f"Node {rank}: Will be killed in {delay_before_kill} seconds")
            
            # Execute broadcast operation
            result = comm.broadcast(data)
            
            # Verify result
            if not will_be_killed:  # Only check nodes that aren't killed
                if use_numpy:
                    success = result is not None and result.shape == (10, 10)
                else:
                    success = result is not None and len(result) == 100
                
                print(f"Node {rank}: Broadcast {'succeeded' if success else 'failed'}")
            
        elif operation == "reduce":
            # Create test data
            if use_numpy:
                data = np.ones((10, 10)) * rank  # Different value for each node
            else:
                data = [rank] * 100  # Simple list with value equal to node rank
            
            print(f"Node {rank}: Performing reduce operation")
            
            # If node will be killed, set up delayed termination
            if will_be_killed:
                kill_timer = mp.Process(target=lambda: (time.sleep(delay_before_kill), os._exit(0)))
                kill_timer.daemon = True
                kill_timer.start()
                print(f"Node {rank}: Will be killed in {delay_before_kill} seconds")
            
            # Execute reduce operation
            result = comm.reduce(data)
            
            # Verify result (at root node)
            if rank == 0 and not will_be_killed:
                if use_numpy:
                    # Calculate expected result: sum of all surviving nodes
                    expected_sum = sum(i for i in range(size) if i not in kill_ranks or i == 0)
                    expected = np.ones((10, 10)) * expected_sum
                    success = result is not None and np.allclose(result, expected)
                else:
                    expected_sum = sum(i for i in range(size) if i not in kill_ranks or i == 0)
                    expected = [expected_sum] * 100
                    success = result is not None and result == expected
                
                print(f"Node {rank}: Reduce {'succeeded' if success else 'failed'}")
                if not success and result is not None:
                    print(f"Expected: {expected[0]}, Got: {result[0]}")
        
        # Ensure all nodes complete the operation
        time.sleep(5)
        
        # Close communicator
        comm.close()
        print(f"Node {rank}: Test completed")
        
    except Exception as e:
        print(f"Node {rank} error: {operation} test failed on node {rank}: {str(e)}")
        print(f"Stack trace: {logging.traceback.format_exc()}")
        return False
    
    return True

def run_test(size, topology, operation, num_kills=1, delay_before_kill=1.0, use_numpy=True):
    """Run fault tolerance test"""
    processes = []
    
    # Randomly select nodes to kill (except root node)
    kill_ranks = random.sample(range(1, size), num_kills) if num_kills > 0 else []
    
    logger.info(f"Starting test with {size} nodes, {topology} topology, testing {operation}")
    logger.info(f"Will kill nodes {kill_ranks} after {delay_before_kill} seconds")
    
    start_time = time.time()
    
    # Start all processes
    for rank in range(size):
        p = mp.Process(target=worker, args=(rank, size, topology, operation, kill_ranks, delay_before_kill, use_numpy))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join(timeout=30)  # Set timeout to prevent infinite blocking
    
    # Check if all processes completed normally
    success = all(not p.is_alive() for p in processes)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    logger.info(f"Test completed in {elapsed:.2f} seconds with {'success' if success else 'failure'}")
    
    return success, elapsed

def main():
    parser = argparse.ArgumentParser(description='Test fault tolerance of collective operations')
    parser.add_argument('--size', type=int, default=8, help='Number of nodes')
    parser.add_argument('--topology', type=str, default='tree', help='Topology to use')
    parser.add_argument('--operation', type=str, default='broadcast', help='Operation to test (broadcast, reduce)')
    parser.add_argument('--kills', type=int, default=1, help='Number of nodes to kill during test')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay before killing nodes (seconds)')
    parser.add_argument('--iterations', type=int, default=5, help='Number of test iterations')
    
    args = parser.parse_args()
    
    print(f"Starting fault tolerance test with:")
    print(f"- {args.size} nodes")
    print(f"- {args.topology} topology")
    print(f"- Testing {args.operation} operation")
    print(f"- Killing {args.kills} nodes after {args.delay} seconds")
    print(f"- Running {args.iterations} iterations")
    
    results = []
    success_count = 0
    
    for i in range(args.iterations):
        print(f"\nIteration {i+1}/{args.iterations}")
        success, elapsed = run_test(args.size, args.topology, args.operation, args.kills, args.delay)
        results.append((success, elapsed))
        if success:
            success_count += 1
    
    # Output results summary
    print("\nTest Results Summary:")
    print(f"Operation: {args.operation}")
    print(f"Topology: {args.topology}")
    print(f"Node count: {args.size}")
    print(f"Killed nodes: {args.kills}")
    print(f"Success rate: {success_count}/{args.iterations} ({success_count/args.iterations*100:.1f}%)")
    
    # Calculate average execution time
    successful_times = [elapsed for success, elapsed in results if success]
    if successful_times:
        avg_time = sum(successful_times) / len(successful_times)
        print(f"Average execution time (successful runs): {avg_time:.2f} seconds")
    
    # Plot results
    iterations = list(range(1, args.iterations + 1))
    execution_times = [elapsed for _, elapsed in results]
    statuses = ["Success" if success else "Failed" for success, _ in results]
    
    plt.figure(figsize=(10, 6))
    
    # Use different colors for success and failure
    for i, (status, time) in enumerate(zip(statuses, execution_times)):
        color = 'green' if status == "Success" else 'red'
        plt.bar(i+1, time, color=color)
    
    plt.xlabel('Iteration')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Fault Tolerance Test: {args.operation} with {args.kills} node failures')
    plt.xticks(iterations)
    
    # Add success/failure labels
    for i, (status, time) in enumerate(zip(statuses, execution_times)):
        plt.text(i+1, time+0.1, status, ha='center')
    
    # Save chart
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"bonus/ft_test_{args.operation}_{args.size}nodes_{args.kills}kills_{timestamp}.png")
    plt.close()

if __name__ == "__main__":
    main() 