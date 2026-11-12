import heapq
import math
import random as _random

import numpy as np

import maze_helper as mh

# --------------------------------------------------------------------------- cấu hình
MOVES = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
_order = list("NESW")
_random_order = False

show_maze = mh.show_maze
animate_maze = mh.animate_maze
MAX_FRAMES = 1500          # giới hạn số khung hình của animation


def set_order(order="NESW", random=False):
    """Thứ tự khám phá các hướng (thêm trạng thái con vào frontier)."""
    global _order, _random_order
    _random_order = bool(random)
    if random:
        print("Directions are checked at every step in random order.")
    else:
        _order = list(order)
        print(f"Directions are checked in the order {_order}")


def _directions():
    d = list(_order)
    if _random_order:
        _random.shuffle(d)
    return d


# --------------------------------------------------------------------------- heuristic
def manhattan(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def euclidean(pos, goal):
    return math.hypot(pos[0] - goal[0], pos[1] - goal[1])


def chebyshev(pos, goal):
    return max(abs(pos[0] - goal[0]), abs(pos[1] - goal[1]))


heuristic = manhattan      # notebook gán lại: ts.heuristic = ts.manhattan / ts.euclidean


def min_index(values):
    """Chỉ số của giá trị nhỏ nhất."""
    return int(np.argmin(values))


# --------------------------------------------------------------------------- Node / chuyển trạng thái
class Node:
    def __init__(self, pos, parent, action, cost):
        self.pos = tuple(int(x) for x in pos)
        self.parent = parent
        self.action = action
        self.cost = cost

    def __str__(self):
        return f"Node - pos = {self.pos}; action = {self.action}; cost = {self.cost}"

    def get_path_from_root(self):
        node, path = self, [self]
        while node.parent is not None:
            node = node.parent
            path.append(node)
        path.reverse()
        return path


def _expand(maze, node):
    """Hàm chuyển trạng thái: các nút con hợp lệ, theo thứ tự hướng hiện tại."""
    for a in _directions():
        dr, dc = MOVES[a]
        r, c = node.pos[0] + dr, node.pos[1] + dc
        if 0 <= r < maze.shape[0] and 0 <= c < maze.shape[1] and maze[r, c] != "X":
            yield Node((r, c), node, a, node.cost + 1)


def _in_path(node, pos):
    """Trạng thái pos đã có trong đường đi từ gốc tới node chưa?"""
    while node is not None:
        if node.pos == pos:
            return True
        node = node.parent
    return False


def _find(maze, what):
    if not (maze == what).any():
        raise ValueError(f"Mê cung không có ô '{what}'")
    p = mh.find_pos(maze, what)
    return (int(p[0]), int(p[1]))


def _frame(maze, explored=(), frontier=(), path=()):
    """Một khung hình animation: vẽ trên BẢN SAO của mê cung."""
    m = np.copy(maze)
    for p in explored:
        if m[p] == " ":
            m[p] = "."
    for p in frontier:                       
        if m[p] in (" ", "."):
            m[p] = "F"
    for p in path:
        if m[p] not in "SG":
            m[p] = "P"
    return m


def _thin(frames):
    if len(frames) <= MAX_FRAMES:
        return frames
    step = int(math.ceil(len(frames) / MAX_FRAMES))
    return frames[::step] + [frames[-1]]


def _result(path_node, reached, extra=None, frames=None):
    path = path_node.get_path_from_root() if path_node is not None else None
    res = {"path": path,
           "actions": [n.action for n in path[1:]] if path else None,
           "reached": set(reached)}
    if frames is not None:
        res["maze_anim"] = _thin(frames)
    if extra:
        res.update(extra)
    return res


# --------------------------------------------------------------------------- BFS / GBFS / A* (Best-first search)
def best_first_search(maze, strategy="BFS", W=1, debug=False, vis=False, animation=False):
   
    start, goal = _find(maze, "S"), _find(maze, "G")

    def f(n):
        if strategy in ("BFS", "UCS"):           # chi phí mỗi bước = 1 nên UCS = BFS
            return n.cost
        if strategy == "DFS":                    # DFS với reached: nút sâu nhất trước (LIFO)
            return -n.cost
        h = heuristic(n.pos, goal)
        if strategy == "GBFS":
            return h
        if strategy == "A*":
            return n.cost + W * h
        raise ValueError(f"Unknown strategy {strategy}")

    fifo = strategy in ("BFS", "UCS")
    root = Node(start, None, None, 0)
    counter = 0
    heap = [(f(root), 0, root)]
    in_frontier = {root.pos: root}
    reached = {root.pos: root}
    expanded = set()
    frames = [_frame(maze, frontier=[root.pos])] if animation else None
    stats = {"expanded": 0, "max_frontier": 1}

    def done(node):
        if animation:
            frames.append(_frame(maze, expanded, in_frontier, [n.pos for n in node.get_path_from_root()]))
        res = _result(node, expanded | set(in_frontier), stats, frames)
        res["frontier"] = set(in_frontier)
        if vis:
            show_path(maze, res)
        return res

    while heap:
        _, _, node = heapq.heappop(heap)
        if in_frontier.get(node.pos) is not node:       # nút cũ đã bị thay bằng nút rẻ hơn
            continue
        del in_frontier[node.pos]
        if debug:
            print(f"expand {node}  f={f(node):.2f}")
        expanded.add(node.pos)
        stats["expanded"] += 1
        for child in _expand(maze, node):
            if child.pos == goal:                        # kiểm tra đích khi sinh nút
                return done(child)
            old = reached.get(child.pos)
            if old is None or (strategy != "DFS" and child.cost < old.cost):
                reached[child.pos] = child
                in_frontier[child.pos] = child
                counter += 1
                heapq.heappush(heap, (f(child), counter if fifo else -counter, child))
        stats["max_frontier"] = max(stats["max_frontier"], len(in_frontier))
        if animation:   # đường đỏ = đường từ S tới nút vừa mở rộng (có ở MỌI khung hình, như bản gốc)
            frames.append(_frame(maze, expanded, in_frontier, [n.pos for n in node.get_path_from_root()]))
    res = _result(None, expanded | set(in_frontier), stats, frames)
    res["frontier"] = set()
    return res


# --------------------------------------------------------------------------- DFS (tree search, không dùng reached)
def DFS(maze, check_cycle=True, limit=None, frontier_option=1, max_tries=100000,
        vis=False, debug_reached=False, animation=False):
    
    start, goal = _find(maze, "S"), _find(maze, "G")
    root = Node(start, None, None, 0)
    frontier = [root]
    in_frontier = {root.pos}
    explored = set()
    frames = [_frame(maze, frontier=[root.pos])] if animation else None
    stats = {"tries": 0, "expanded": 0, "max_frontier": 1, "cutoff": False}

    def finish(node):
        mem = set()
        if debug_reached:      # reached = đã mở rộng + còn trong frontier + các ô trên đường đi (khớp output của giảng viên)
            mem = explored | {n.pos for n in frontier} | ({n.pos for n in node.get_path_from_root()} if node else set())
        res = _result(node, mem, stats, frames)
        res["frontier"] = {n.pos for n in frontier} if debug_reached else set()
        if vis:
            show_path(maze, res)
        return res

    while frontier and stats["tries"] < max_tries:
        stats["tries"] += 1
        node = frontier.pop()
        in_frontier.discard(node.pos)

        if node.pos == goal:
            if animation:
                frames.append(_frame(maze, frontier=[n.pos for n in frontier],
                                     path=[n.pos for n in node.get_path_from_root()]))
            return finish(node)

        if check_cycle and _in_path(node.parent, node.pos):      # Chu trình: bỏ qua
            continue
        if limit is not None and node.cost >= limit:             # chạm giới hạn độ sâu
            stats["cutoff"] = True
            continue

        explored.add(node.pos)
        stats["expanded"] += 1
        for child in _expand(maze, node):
            if frontier_option == 2 and child.pos in in_frontier:      # đã có trong frontier -> bỏ
                continue
            if frontier_option == 1 and child.pos in in_frontier:      # xóa mục cũ, thêm mục mới lên ĐỈNH stack
                for i, old in enumerate(frontier):
                    if old.pos == child.pos:
                        del frontier[i]
                        break
            frontier.append(child)
            in_frontier.add(child.pos)
        stats["max_frontier"] = max(stats["max_frontier"], len(frontier))

        if animation:   # bộ nhớ hiện tại: đường đi hiện tại (đỏ) + frontier (cam)
            cur = [n.pos for n in node.get_path_from_root()]
            frames.append(_frame(maze, frontier=[n.pos for n in frontier], path=cur))

    return finish(None)


# --------------------------------------------------------------------------- IDS
def IDS(maze, frontier_option=2, max_tries=100000, vis=False, check_cycle=True):
   
    n_free = int((maze != "X").sum())
    res = None
    for limit in range(0, min(max_tries, n_free + 1)):
        res = DFS(maze, check_cycle=check_cycle, limit=limit, frontier_option=frontier_option,
                  max_tries=float("inf"), vis=False, debug_reached=False)
        if res["path"] is not None:
            break
        if not res["cutoff"]:           # không nhánh nào bị cắt bởi limit: không có lời giải
            break
    res["limit"] = limit
    if vis:
        show_path(maze, res)
    return res


# --------------------------------------------------------------------------- hiển thị
def show_path(maze, result, fontsize=10):
    if result["path"] is not None:
        print(f"Path length: {len(result['path']) - 1}")
    reached = result.get("reached") or set()
    print(f"Reached squares: {len(reached)}")
    path = [n.pos for n in result["path"]] if result["path"] else []
    m = _frame(maze, explored=reached, frontier=result.get("frontier", ()), path=path)
    mh.show_maze(m, fontsize=fontsize)


set_order("NESW")      # mặc định; in dòng thông báo ngay khi import (như bản gốc)
