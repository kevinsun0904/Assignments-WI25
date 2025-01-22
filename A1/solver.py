import math
import random
from collections import deque, defaultdict
import heapq
import numpy as np

random.seed(42)

###############################################################################
#                                Node Class                                   #
###############################################################################

class Node:
    """
    Represents a graph node with an undirected adjacency list.
    'value' can store (row, col), or any unique identifier.
    'neighbors' is a list of connected Node objects (undirected).
    """
    def __init__(self, value):
        self.value = value
        self.neighbors = []

    def add_neighbor(self, node):
        """
        Adds an undirected edge between self and node:
         - self includes node in self.neighbors
         - node includes self in node.neighbors (undirected)
        """
        self.neighbors.append(node)

    def __repr__(self):
        return f"Node({self.value})"
    
    def __lt__(self, other):
        return self.value < other.value


###############################################################################
#                   Maze -> Graph Conversion (Undirected)                     #
###############################################################################

def parse_maze_to_graph(maze):
    """
    Converts a 2D maze (numpy array) into an undirected graph of Node objects.
    maze[r][c] == 0 means open cell; 1 means wall/blocked.

    Returns:
        nodes_dict: dict[(r, c): Node] mapping each open cell to its Node
        start_node : Node corresponding to (0, 0), or None if blocked
        goal_node  : Node corresponding to (rows-1, cols-1), or None if blocked
    """
    rows, cols = maze.shape
    nodes_dict = {}

    # 1) Create a Node for each open cell
    # 2) Link each node with valid neighbors in four directions (undirected)
    # 3) Identify start_node (if (0,0) is open) and goal_node (if (rows-1, cols-1) is open)

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    for row in range(rows):
        for col in range(cols):
            if maze[row][col] == 0:
                nodes_dict[(row, col)] = Node((row, col))

    for index in nodes_dict:
            for direction in directions:
                new_index = (index[0] + direction[0], index[1] + direction[1])
                if new_index in nodes_dict:
                    nodes_dict[index].neighbors.append(nodes_dict[new_index])


    start_node = nodes_dict[(0, 0)] if (0, 0) in nodes_dict else None
    goal_node = nodes_dict[(rows - 1, cols - 1)] if (rows - 1, cols - 1) in nodes_dict else None

    return nodes_dict, start_node, goal_node


###############################################################################
#                         BFS (Graph-based)                                    #
###############################################################################

def bfs(start_node, goal_node):
    """
    Breadth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a queue (collections.deque) to hold nodes to explore.
      2. Track visited nodes so you don’t revisit.
      3. Also track parent_map to reconstruct the path once goal_node is reached.
    """
    
    queue = deque()
    visited = set()
    parent_map = {}
    parent_map[start_node] = None
    queue.append(start_node)
    visited.add(start_node)

    while len(queue) != 0:
        node = queue.popleft()

        if node == goal_node:
            return reconstruct_path(goal_node, parent_map)
        
        for neighbor in node.neighbors:
            if neighbor not in visited:
                parent_map[neighbor] = node
                queue.append(neighbor)
                visited.add(neighbor)

    return None


###############################################################################
#                          DFS (Graph-based)                                   #
###############################################################################

def dfs(start_node, goal_node):
    """
    Depth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a stack (Python list) to hold nodes to explore.
      2. Keep track of visited nodes to avoid cycles.
      3. Reconstruct path via parent_map if goal_node is found.
    """
    
    stack = [start_node]
    visited = set()
    parent_map = {}
    parent_map[start_node] = None
    visited.add(start_node)

    while len(stack) != 0:
        node = stack.pop()

        if node == goal_node:
            return reconstruct_path(goal_node, parent_map)
        
        for neighbor in node.neighbors:
            if neighbor not in visited:
                parent_map[neighbor] = node
                stack.append(neighbor)
                visited.add(neighbor)

    return None


###############################################################################
#                    A* (Graph-based with Manhattan)                           #
###############################################################################

def astar(start_node, goal_node):
    """
    A* search on an undirected graph of Node objects.
    Uses manhattan_distance as the heuristic, assuming node.value = (row, col).
    Returns a path (list of (row, col)) or None if not found.

    Steps (suggested):
      1. Maintain a min-heap/priority queue (heapq) where each entry is (f_score, node).
      2. f_score[node] = g_score[node] + heuristic(node, goal_node).
      3. g_score[node] is the cost from start_node to node.
      4. Expand the node with the smallest f_score, update neighbors if a better path is found.
    """
    
    heap = []
    g_score = {}
    f_score = {}
    parent_map = {}
    parent_map[start_node] = None
    visited = set()
    heapq.heappush(heap, (manhattan_distance(start_node, goal_node), start_node))
    g_score[start_node] = 0
    visited.add(start_node)

    while len(heap) != 0:
        score, node = heapq.heappop(heap)

        if node == goal_node:
            return reconstruct_path(goal_node, parent_map)
        
        for neighbor in node.neighbors:
            if neighbor not in visited:
                parent_map[neighbor] = node
                g_score[neighbor] = g_score[node] + 1
                # f_score[neighbor] shouldn't change? Since the edge weights are all 1
                f_score[neighbor] = min(f_score.get(neighbor, float('inf')), g_score[neighbor] + manhattan_distance(neighbor, goal_node))
                heapq.heappush(heap, (f_score[neighbor], neighbor))
                visited.add(neighbor)

    return None

def manhattan_distance(node_a, node_b):
    """
    Helper: Manhattan distance between node_a.value and node_b.value 
    if they are (row, col) pairs.
    """

    return abs(node_a.value[0] - node_b.value[0]) + abs(node_a.value[1] - node_b.value[1])


###############################################################################
#                 Bidirectional Search (Graph-based)                          #
###############################################################################

def bidirectional_search(start_node, goal_node):
    """
    Bidirectional search on an undirected graph of Node objects.
    Returns list of (row, col) from start to goal, or None if not found.

    Steps (suggested):
      1. Maintain two frontiers (queues), one from start_node, one from goal_node.
      2. Alternate expansions between these two queues.
      3. If the frontiers intersect, reconstruct the path by combining partial paths.
    """

    queue = [deque(), deque()]
    visited = [set(), set()]
    parent = [{}, {}]
    visited[0].add(start_node)
    visited[1].add(goal_node)
    parent[0][start_node] = None
    parent[1][goal_node] = None
    queue[0].append(start_node)
    queue[1].append(goal_node)

    direction = 0
    while len(queue[0]) > 0 and len(queue[1]) > 0:
        other_direction = (direction + 1) % 2
        node = queue[direction].popleft()

        if node in visited[other_direction]:
            to_list = reconstruct_path(node, parent[direction])
            to_list.pop()
            from_list = reconstruct_path(node, parent[other_direction])
            from_list.reverse()
            return to_list + from_list
        
        for neighbor in node.neighbors:
            if neighbor not in visited[direction]:
                parent[direction][neighbor] = node
                queue[direction].append(neighbor)
                visited[direction].add(neighbor)

        direction = other_direction

    return None


###############################################################################
#             Simulated Annealing (Graph-based)                               #
###############################################################################

def simulated_annealing(start_node, goal_node, temperature=1.0, cooling_rate=0.99, min_temperature=0.01):
    """
    A basic simulated annealing approach on an undirected graph of Node objects.
    - The 'cost' is the manhattan_distance to the goal.
    - We randomly choose a neighbor and possibly move there.
    Returns a list of (row, col) from start to goal (the path traveled), or None if not reached.

    Steps (suggested):
      1. Start with 'current' = start_node, compute cost = manhattan_distance(current, goal_node).
      2. Pick a random neighbor. Compute next_cost.
      3. If next_cost < current_cost, move. Otherwise, move with probability e^(-cost_diff / temperature).
      4. Decrease temperature each step by cooling_rate until below min_temperature or we reach goal_node.
    """
    
    current_node = start_node
    current_cost = manhattan_distance(start_node,  goal_node)
    path = [start_node.value]

    while temperature > min_temperature:
        if current_node == goal_node:
            return path
        
        next_node = random.choice(current_node.neighbors)
        next_cost = manhattan_distance(next_node, goal_node)

        cost_diff = next_cost - current_cost

        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current_node = next_node
            current_cost = next_cost
            path.append(current_node.value)

        temperature *= cooling_rate

    return None


###############################################################################
#                           Helper: Reconstruct Path                           #
###############################################################################

def reconstruct_path(end_node, parent_map):
    """
    Reconstructs a path by tracing parent_map up to None.
    Returns a list of node.value from the start to 'end_node'.

    'parent_map' is typically dict[Node, Node], where parent_map[node] = parent.

    Steps (suggested):
      1. Start with end_node, follow parent_map[node] until None.
      2. Collect node.value, reverse the list, return it.
    """
    path = []
    curr = end_node
    while curr != None:
        path.append(curr.value)
        curr = parent_map[curr]
    path.reverse()
    return path


###############################################################################
#                              Demo / Testing                                 #
###############################################################################
if __name__ == "__main__":
    # A small demonstration that the code runs (with placeholders).
    # This won't do much yet, as everything is unimplemented.
    random.seed(42)
    np.random.seed(42)

    # Example small maze: 0 => open, 1 => wall
    maze_data = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ])

    # Parse into an undirected graph
    nodes_dict, start_node, goal_node = parse_maze_to_graph(maze_data)
    print("Created graph with", len(nodes_dict), "nodes.")
    print("Start Node:", start_node)
    print("Goal Node :", goal_node)

    # Test BFS (will return None until implemented)
    path_bfs = bfs(start_node, goal_node)
    print("BFS Path:", path_bfs)

    # Similarly test DFS, A*, etc.
    # path_dfs = dfs(start_node, goal_node)
    # path_astar = astar(start_node, goal_node)
    # ...
