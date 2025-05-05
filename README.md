[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/k1bgkW6c)
# **CSC4303 - Assignment 6: Collective Communication**  

**Due Date**:  4.30 23:59 (UTC+8)    
**Total Points**: 12 + 4 (Bonus)  
**Name**: Shi Shuhao
**ID**: 122090466


## **Objective**  
Implement and benchmark **collective communication** operations (Broadcast, Reduce, Scatter, Gather, AllReduce, etc.) using different network topologies (Ring, Tree). Analyze their performance under varying node counts and data sizes.

---
## Project Architecture

The project follows a modular design with the following components:

### Core Components

- NetworkLayer
    - Low-level network communication module for message passing between nodes

- Communicator 
  - Communication interface that integrates network layer and topology

- Topology
  - Abstract topology class and implementations:
    - **RingTopology**: Ring-based communication pattern
    - **TreeTopology**: Tree topology

- Collective Communication Primitives
  - **broadcast**: Distribute data from root to all processes
  - **scatter**: Distribute different portions of data to different processes
  - **gather**: Collect data from all processes to root
  - **reduce**: Combine data from all processes to root using an operation
  - **allgather**: Gather data from all processes and distribute to all
  - **allreduce**: Reduce data and distribute results to all processes
  - **reduce_scatter**: Reduce data and scatter results to all processes

---

## **Requirements**  

### **1. Implementation (10 points)**  
Implement the following in Python:  

#### **(a) Core Components**  


- **`topology/`**:  
  - Topology and primitives are disaggregated in this codebase, you may either follow the codebase or implement your own aggregated primitives and topologies. This won't affect the grading.
  - Concrete implementations:  
    - `RingTopology`  
    - `TreeTopology`

- **`collective/`**:  
  - Implement **at least 5** collective operations (despite broadcast is already implemented), using ring and tree topologies for each operation (if some operation is not applicable to a specific topology, you can argue in the report and use other topologies):   
    - `allreduce.py`  
    - `reduce_scatter.py`
    - `scatter.py`
    - `gather.py`
    - `reduce.py`  
    - `allgather.py`  

#### **(b) Benchmarking and Summary (2 points)**  
- Write a simple summary `report.md` with:  
  - Performance Analysis: 
    - Tables/plots comparing benchmarking under **varying node counts** (e.g., 4, 6, 8, 16) and  **varying data sizes** (e.g., 0.5, 1KB, 2KB, 4KB, 16KB)
  - Discussion 
    - Which topology is best for each operation? Why?  

---
### Bonus (4 point)
- For extra credit, implement and evaluate performance optimizations on fault tolerance:
    - e.g., one node failed during communication, setup a new node to recover and continue the communication
    - May use heart beat mechanism to detect node failure
    - Support at least 2 collective communication primtives by implementing required interfaces
    - Write extra fault tolerance test script and give analysis
    - We will test fault tolerance by randomly killing nodes during communication
- Requirements:
    - Create a /bonus directory containing all code for your optimized system. This should be a separate implementation; do not modify your original submission.
The /bonus directory must also include a report.md file detailing your optimization strategies and presenting benchmark results obtained by running benchmark.py comparing the original and optimized systems.
Instructions for running your optimized system must also be included in the report.md.
Bonus points will be added to your assignment score, but the total score for all assignments (including bonus points) cannot exceed 60 points (60% of the final grade).



