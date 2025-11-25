
### II. Conceptual Improvements and Justification for Fixes

While the overall structure is correct, certain implementation choices needed refinement to ensure the algorithms operate reliably across all edge cases of their defined domains.

#### 1. Ensuring Correctness in the Face of Negative Weights

**Conceptual Issue:** **Dijkstra's** and **A\*** algorithms rely on the assumption that edge weights are **non-negative**. If a negative edge exists, the fundamental principle that a node's distance is final upon extraction from the heap is violated.

**Justification for the Fix (Early Detection):**
* The original implementation attempted to handle negative weights by simply skipping the relaxation of a negative edge (`continue`). This is insufficient because a negative edge elsewhere could have compromised the distances of nodes already in the heap.
* The fix—**pre-checking the entire graph for negative edges** and raising an error or status change—is necessary to uphold the algorithms' **domain constraints**. It prevents the execution from producing an unreliable or suboptimal result when the input violates the core assumption of the algorithm.

#### 2. Streamlining the A\* Heap Management 
```
# short_path_finding.py: astar_with_frontier

# Remove:
# in_open: Dict[Hashable, bool] = {node: False for node in G.nodes}
# in_open[source] = True

while open_heap:
f, u = heappop(open_heap)

    #Substitute check "in_open" with a g-score check
    # If f is greater than the current g-score + heuristics (with float tolerance),
    # it is an expired item.
    current_f = g[u] + heuristic(u, target)
    if f > current_f + 1e-9: 
        continue # Stale entry

    frontier.append(u)

    # ...
    # remove 'in_open[v] = True' and 'in_open[u] = False'
```
**Conceptual Issue:** When a node's $g$-score (actual cost from source) is improved, it is re-added to the priority queue. This creates **stale entries** (outdated paths) in the heap, which can reduce efficiency if they are processed.

**Justification for the Fix (Standard Heap Check):**
* The initial approach used a separate dictionary (`in_open`) to track if an entry was still "fresh," which added overhead.
* The improved implementation aligns with the standard and most efficient way to manage a heap in Dijkstra-like algorithms: **checking the extracted value against the current best known value** (the current $g$-score in the `g` dictionary, combined with $h$).
* By comparing the extracted $f$-score with the current minimum $g(u) + h(u)$, one can reliably discard stale entries without maintaining an extra tracking data structure, thereby improving **code clarity** and **run-time efficiency**. 