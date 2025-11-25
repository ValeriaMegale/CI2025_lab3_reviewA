import math
from heapq import heappush, heappop
from typing import Callable, Dict, Hashable, Tuple, List, Any
import numpy as np
import networkx as nx

# Graph construction
def build_graph_from_matrix(cost_matrix: np.ndarray) -> nx.DiGraph:
    """
    Turn a cost matrix into a directed NetworkX graph.
    cost_matrix[i, j] = weight of edge i -> j, or np.inf if no edge.
    """
    n = cost_matrix.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = float(cost_matrix[i, j])
            if math.isfinite(w):
                G.add_edge(i, j, weight=w)

    return G

# Heuristic for A* function
def euclidean_heuristic_factory(
    coords: np.ndarray,
    scale: float = 1.0,
) -> Callable[[Hashable, Hashable], float]:
    """
    Given coordinates for nodes, return an admissible heuristic for A*.

    coords: shape (n, 2), coords[i] = (x_i, y_i)
    scale:  multiply distances by this factor. If your edge weights are
            roughly 'distance * factor + noise', pick that factor.
    """
    def h(u: int, v: int) -> float:
        dx = coords[u, 0] - coords[v, 0]
        dy = coords[u, 1] - coords[v, 1]
        return scale * math.hypot(dx, dy)

    return h

# Helper functions
def _reconstruct_path(pred: Dict[Any, Any], source: Any, target: Any) -> List[Any]:
    # If source and target are the same, return single-node path
    if source == target:
        return [source]
    # If target is not reachable, return empty path
    if target not in pred:
        return []
    # Reconstruct path from target back to source
    path = [target]
    cur = target
    while cur != source:
        cur = pred.get(cur)
        if cur is None:
            return []
        path.append(cur)
    path.reverse()
    return path

# Single-pair variants with frontier, cost, best path, status
def dijkstra_with_frontier(
    G: nx.DiGraph,
    source: Hashable,
    target: Hashable,
) -> Dict[str, Any]:
    """
    Dijkstra between source and target.

    Returns a dict with:
        - 'frontier': list of nodes expanded (in the order they are popped)
        - 'nodes_explored': number of unique nodes explored
        - 'cost': best distance (float or math.inf)
        - 'path': list of nodes from source to target
        - 'status': 'ok' or 'no_path'
    """
    # Handle source == target case
    if source == target:
        return {
            "frontier": [source],
            "nodes_explored": 1,
            "cost": 0.0,
            "path": [source],
            "status": "ok",
        }
    
    dist: Dict[Hashable, float] = {node: math.inf for node in G.nodes}
    dist[source] = 0.0
    pred: Dict[Hashable, Hashable] = {}

    frontier: List[Hashable] = []
    heap: List[Tuple[float, Hashable]] = [(0.0, source)]

    # Pre-check for negative weights
    if any(data.get("weight", 1.0) < 0 for u, v, data in G.edges(data=True)):
        raise ValueError("Dijkstra's algorithm does not support negative edge weights.")
    while heap:
        d, u = heappop(heap)
        if d > dist[u]:
            continue
        frontier.append(u)

        if u == target:
            break

        for v, data in G[u].items():
            w = float(data.get("weight", 1.0))
            if w < 0:
                # Dijkstra is not valid with negative edges
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heappush(heap, (nd, v))

    cost = dist[target]
    nodes_explored = len(frontier)
    if not math.isfinite(cost):
        return {
            "frontier": frontier,
            "nodes_explored": nodes_explored,
            "cost": math.inf,
            "path": [],
            "status": "no_path",
        }

    path = _reconstruct_path(pred, source, target)
    return {
        "frontier": frontier,
        "nodes_explored": nodes_explored,
        "cost": cost,
        "path": path,
        "status": "ok",
    }


def bellman_ford_with_frontier(
    G: nx.DiGraph,
    source: Hashable,
    target: Hashable,
) -> Dict[str, Any]:
    """
    Bellman–Ford between source and target.

    Returns a dict with:
        - 'frontier': list of nodes whose distance was improved (may have duplicates)
        - 'nodes_explored': number of unique nodes explored
        - 'cost': best distance (float or math.inf)
        - 'path': list of nodes from source to target (if well-defined)
        - 'status': 'ok', 'no_path', or 'negative_cycle'
    """
    # Handle source == target case
    if source == target:
        return {
            "frontier": [],
            "nodes_explored": 1,
            "cost": 0.0,
            "path": [source],
            "status": "ok",
        }
    
    nodes = list(G.nodes)
    dist: Dict[Hashable, float] = {node: math.inf for node in nodes}
    dist[source] = 0.0
    pred: Dict[Hashable, Hashable] = {}

    frontier: List[Hashable] = []
    explored: set = {source}  # Track unique nodes explored

    # Relax edges up to |V|-1 times
    for _ in range(len(nodes) - 1):
        updated = False
        for u, v, data in G.edges(data=True):
            w = float(data.get("weight", 1.0))
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                frontier.append(v)
                explored.add(v)  # Track unique nodes
                updated = True
        if not updated:
            break

    # Check for negative cycles
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        if dist[u] + w < dist[v]:
            return {
                "frontier": frontier,
                "nodes_explored": len(explored),
                "cost": math.nan,
                "path": [],
                "status": "negative_cycle",
            }

    cost = dist[target]
    nodes_explored = len(explored)
    if not math.isfinite(cost):
        return {
            "frontier": frontier,
            "nodes_explored": nodes_explored,
            "cost": math.inf,
            "path": [],
            "status": "no_path",
        }

    path = _reconstruct_path(pred, source, target)
    return {
        "frontier": frontier,
        "nodes_explored": nodes_explored,
        "cost": cost,
        "path": path,
        "status": "ok",
    }

def astar_with_frontier(
    G: nx.DiGraph,
    source: Hashable,
    target: Hashable,
    heuristic: Callable[[Hashable, Hashable], float],
) -> Dict[str, Any]:
    """
    A* between source and target.

    Returns a dict with:
        - 'frontier': list of nodes expanded (popped from open set)
        - 'nodes_explored': number of unique nodes explored
        - 'cost': best distance (float or math.inf)
        - 'path': list of nodes from source to target
        - 'status': 'ok' or 'no_path'

    Assumes NON-NEGATIVE edge weights and an admissible heuristic.
    """
    # Handle source == target case
    if source == target:
        return {
            "frontier": [source],
            "nodes_explored": 1,
            "cost": 0.0,
            "path": [source],
            "status": "ok",
        }
    
    open_heap: List[Tuple[float, Hashable]] = []
    heappush(open_heap, (0.0, source))

    g: Dict[Hashable, float] = {node: math.inf for node in G.nodes}
    g[source] = 0.0

    pred: Dict[Hashable, Hashable] = {}

    frontier: List[Hashable] = []
    in_open: Dict[Hashable, bool] = {node: False for node in G.nodes}
    in_open[source] = True

    while open_heap:
        f, u = heappop(open_heap)
        if not in_open[u]:
            continue  # stale entry
        in_open[u] = False

        frontier.append(u)

        if u == target:
            path = _reconstruct_path(pred, source, target)
            nodes_explored = len(frontier)
            return {
                "frontier": frontier,
                "nodes_explored": nodes_explored,
                "cost": g[target],
                "path": path,
                "status": "ok",
            }

        for v, data in G[u].items():
            w = float(data.get("weight", 1.0))
            if w < 0:
                # A* is not valid with negative edges
                continue
            tentative_g = g[u] + w
            if tentative_g < g[v]:
                g[v] = tentative_g
                pred[v] = u
                f_score = tentative_g + heuristic(v, target)
                heappush(open_heap, (f_score, v))
                in_open[v] = True

    # No path
    nodes_explored = len(frontier)
    return {
        "frontier": frontier,
        "nodes_explored": nodes_explored,
        "cost": math.inf,
        "path": [],
        "status": "no_path",
    }

# All-pairs variants (for global comparison)
def all_pairs_dijkstra(
    G: nx.DiGraph,
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, List[int]]]]:
    """
    Run Dijkstra from every node (assumes NON-NEGATIVE edge weights).

    Returns:
    costs[source][target] = shortest distance
    paths[source][target] = list of nodes along the path
    """
    costs: Dict[int, Dict[int, float]] = {}
    paths: Dict[int, Dict[int, List[int]]] = {}

    for s in G.nodes:
        length, path = nx.single_source_dijkstra(G, source=s, weight="weight")
        costs[s] = length          # dict target -> distance
        paths[s] = path            # dict target -> path list

    return costs, paths


def all_pairs_bellman_ford(
    G: nx.DiGraph,
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, List[int]]]]:
    """
    Run Bellman–Ford from every node (handles negative edges, but
    assumes NO negative cycles reachable from each source).

    If a negative cycle exists anywhere in the graph, this function
    raises a ValueError.
    """
    # global check; if there is a negative cycle anywhere, shortest
    # paths may be undefined
    if nx.negative_edge_cycle(G, weight="weight"):
        raise ValueError("Graph contains a negative weight cycle.")

    costs: Dict[int, Dict[int, float]] = {}
    paths: Dict[int, Dict[int, List[int]]] = {}

    for s in G.nodes:
        length, path = nx.single_source_bellman_ford(G, source=s, weight="weight")
        costs[s] = length          # dict target -> distance
        paths[s] = path            # dict target -> path list

    return costs, paths


def all_pairs_astar(
    G: nx.DiGraph,
    heuristic: Callable[[Hashable, Hashable], float],
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, List[int]]]]:
    """
    Run A* between all ordered pairs of distinct nodes.
    This is mainly for COMPARISON; it's O(n^2) A* runs.
    """
    costs: Dict[int, Dict[int, float]] = {u: {} for u in G.nodes}
    paths: Dict[int, Dict[int, List[int]]] = {u: {} for u in G.nodes}

    nodes = list(G.nodes)
    for s in nodes:
        for t in nodes:
            if s == t:
                continue
            try:
                p = nx.astar_path(G, s, t, heuristic=heuristic, weight="weight")
                d = nx.astar_path_length(G, s, t, heuristic=heuristic, weight="weight")
                costs[s][t] = float(d)
                paths[s][t] = p
            except nx.NetworkXNoPath:
                # Just skip unreachable pairs
                continue

    return costs, paths
    
# Utility: find global best positive path function  
def best_positive_path(
    costs: Dict[int, Dict[int, float]],
    paths: Dict[int, Dict[int, List[int]]],
):
    """
    Among all pairs, find the (source, target) with the smallest
    strictly positive cost.
    Returns ( (source, target), path, cost ).
    If nothing positive is found, returns ((None, None), [], np.inf).
    """
    best_pair = (None, None)
    best_path: List[int] = []
    best_cost = math.inf

    for s, targets in costs.items():
        for t, d in targets.items():
            if s == t:
                continue
            if d is None or not math.isfinite(d):
                continue
            if d > 0 and d < best_cost:
                best_cost = d
                best_pair = (s, t)
                best_path = paths[s][t]

    return best_pair, best_path, best_cost
