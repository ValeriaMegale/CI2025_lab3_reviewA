# CI2025 Lab 3: Shortest Path Finding Algorithms

This project implements and compares three fundamental shortest path algorithms **POSITIVE VALUE ONLY**: **Dijkstra's Algorithm**, **Bellman-Ford Algorithm**, and **A\* Search Algorithm**. The implementation uses NetworkX for graph representation and provides comprehensive tracking of algorithm behavior including frontier exploration, nodes explored, path costs, and execution status.

## Project Structure

- **`short_path_finding.py`**: Core implementation of the three shortest path algorithms with frontier tracking
- **`lab3_shortest_paths_demo.ipynb`**: Jupyter notebook demonstrating algorithm usage, comparison, and visualization

## Problem Overview

The goal is to find the shortest path between two randomly selected cities (nodes) in a weighted directed graph. The graph is generated with configurable parameters:
- **Size**: Number of cities/nodes (10, 20, 50, 100, 200, 500, 1000)
- **Density**: Probability of edge existence between any two nodes (0.2, 0.5, 0.8, 1.0)
- **Noise Level**: Random variation added to edge weights (0.0, 0.1, 0.5, 0.8)
- **Negative Values**: Whether to allow negative edge weights (affects algorithm selection like bellman ford)

## How the Problem Solving Works

### 1. Graph Generation (`lab3_shortest_paths_demo.ipynb`)

The `create_problem()` function generates a random graph:

1. **City Coordinates**: Each city is assigned random 2D coordinates in [0,1]×[0,1]
2. **Edge Creation**: For each pair of cities (i, j), an edge is created with probability `density`
3. **Edge Weight Calculation**: 
   - Base distance = Euclidean distance between city coordinates
   - Noise is added: `weight = base_distance + noise`
   - If `negative_values=True`, noise can be negative (range: [-noise_level, +noise_level])
   - If `negative_values=False`, noise is non-negative (range: [0, noise_level])
4. **Graph Construction**: The cost matrix is converted to a NetworkX `DiGraph` using `build_graph_from_matrix()`

### 2. City Selection

Two cities are randomly selected using:
- **Dynamic Seeding**: Uses `time.time()` to ensure different results on each run or we can use `rng = np.random.default_rng()`
- **Random Shuffling**: The node list is shuffled for display purposes
- **Unique Selection**: Two distinct nodes are chosen using `rng.choice(nodes, size=2, replace=False)`

### 3. Algorithm Execution

All three algorithms are run on the same source-target pair, and their results are compared:

- **Dijkstra's Algorithm**: Uses a priority queue (min-heap) to explore nodes in order of increasing distance from source
- **Bellman-Ford Algorithm**: Relaxes all edges up to |V|-1 times to find shortest paths
- **A\* Algorithm**: Uses Dijkstra-like exploration guided by a heuristic function (Euclidean distance to target)

## Algorithm Mechanisms

### Dijkstra's Algorithm (`dijkstra_with_frontier`)

**How it works:**
1. Initialize distance to source as 0, all other nodes as ∞
2. Maintain a priority queue (min-heap) of nodes to explore, ordered by distance
3. Repeatedly:
   - Pop the node with minimum distance from the queue
   - Add it to the frontier (list of explored nodes)
   - If it's the target, stop
   - For each neighbor, if a shorter path is found, update distance and add to queue

**Key Characteristics:**
- **Greedy**: Always explores the closest unvisited node first
- **Optimal**: Guarantees shortest path for non-negative edge weights
- **Efficient**: Typically explores fewer nodes than Bellman-Ford
- **Limitation**: Cannot handle negative edge weights

**Time Complexity**: O((V + E) log V)

### Bellman-Ford Algorithm (`bellman_ford_with_frontier`)

**How it works:**
1. Initialize distance to source as 0, all other nodes as ∞
2. Relax all edges up to |V|-1 times:
   - For each edge (u, v) with weight w: if `dist[u] + w < dist[v]`, update `dist[v]`
3. Check for negative cycles: if any edge can still be relaxed, a negative cycle exists
4. Reconstruct path using predecessor array

**Key Characteristics:**
- **Systematic**: Examines all edges in each iteration
- **Robust**: Handles negative edge weights (unlike Dijkstra)
- **Complete**: Can detect negative cycles
- **Less Efficient**: Typically explores more nodes than Dijkstra or A\*

**Time Complexity**: O(V × E)

### A\* Algorithm (`astar_with_frontier`)

**How it works:**
1. Uses a priority queue ordered by `f(n) = g(n) + h(n)` where:
   - `g(n)` = actual cost from source to node n
   - `h(n)` = heuristic estimate from node n to target (Euclidean distance)
2. Similar to Dijkstra, but prioritizes nodes that appear closer to the target
3. The heuristic guides exploration toward the target, reducing unnecessary exploration

**Key Characteristics:**
- **Informed Search**: Uses heuristic to guide exploration
- **Optimal**: Guarantees shortest path if heuristic is admissible (never overestimates)
- **Efficient**: Typically explores fewer nodes than Dijkstra due to heuristic guidance
- **Limitation**: Requires non-negative edge weights and an admissible heuristic

**Time Complexity**: O((V + E) log V) (same as Dijkstra, but with better constant factors)

**In Summary**: Bellman-Ford trades efficiency for generality—it can handle negative weights and detect cycles, but pays the cost of exploring more of the graph systematically.
and the reason other algorithms are faster in this problem because they are better in positive values

##  Algorithm Comparison

| Algorithm | Edge Weights | Negative Cycles | Typical Nodes Explored |
|-----------|--------------|-----------------|----------------------|---------------|
| **Dijkstra** | Non-negative only | Cannot detect | Medium |
| **Bellman-Ford** | Any (including negative) | Can detect | High |
| **A\*** | Non-negative only | Cannot detect | Low |


### Customizing Parameters

In the notebook, modify these parameters in the second cell:

```python
size = 200              # Number of cities
density = 1.0           # Edge probability (0.0 to 1.0)
noise_level = 0.5       # Random noise added to edge weights
negative_values = False # Allow negative edge weights
```
