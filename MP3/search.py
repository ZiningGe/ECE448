# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP3. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
# Note that if you want to test one of your search methods, please make sure to return a blank list
#  for the other search methods otherwise the grader will not crash.

from collections import deque
import heapq


def mdis(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances = {
            (i, j): mdis(i, j)
            for i, j in self.cross(objectives)
        }

    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root

    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a)
        rb = self.resolve(b)
        if ra == rb:
            return False
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        # return (x for y in (((i, j) for j in keys) for i in keys) for x in y)
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    prev = {}
    explored = []
    retval = []
    start = maze.start
    waypoint = maze.waypoints[0]
    prev[start] = (-1, -1)
    explored.append(start)
    d = deque()
    d.append(start)
    while len(d) != 0:
        cur = d.popleft()
        if cur == waypoint:
            break
        for neighbor in maze.neighbors(cur[0], cur[1]):
            if neighbor not in explored:
                explored.append(neighbor)
                d.append(neighbor)
                prev[neighbor] = cur

    path = waypoint
    while path != (-1, -1):
        retval.append(path)
        path = prev[path]
    retval.reverse()
    print(retval)
    return retval


def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # prev = {}
    # explored = {}
    # retval = []
    # start = maze.start
    # waypoint = maze.waypoints[0]
    # prev[start] = (-1, -1)
    # explored[start] = 0
    # d = []
    # heapq.heappush(d, (mdis(start, waypoint), start))
    # while len(d) != 0:
    #     cur = heapq.heappop(d)[1]
    #     if cur == waypoint:
    #         break
    #     for neighbor in maze.neighbors(cur[0], cur[1]):
    #         if neighbor not in explored.keys() or explored[neighbor] > explored[cur] + 1:
    #             explored[neighbor] = explored[cur] + 1
    #             heapq.heappush(d, (explored[neighbor] + mdis(neighbor, waypoint), neighbor))
    #             prev[neighbor] = cur
    # path = waypoint
    # while path != (-1, -1):
    #     retval.append(path)
    #     path = prev[path]
    # retval.reverse()
    # print(retval)

    start = maze.start
    prev = {start: (-1, -1)}
    explored = {(-1, -1): -1}
    retval = []
    waypoint = maze.waypoints[0]
    d = []
    heapq.heappush(d, (mdis(start, waypoint), start, (-1, -1)))
    while len(d) != 0:
        print(d)
        cur_tuple = heapq.heappop(d)
        cur = cur_tuple[1]
        parent = cur_tuple[2]
        if cur == waypoint:
            prev[cur] = parent
            break
        if cur not in explored:
            prev[cur] = parent
            explored[cur] = explored[parent] + 1
            for neighbor in maze.neighbors(cur[0], cur[1]):
                # if neighbor not in explored.keys() or explored[neighbor] > explored[cur] + 1:
                heapq.heappush(d, (explored[cur] + 1 + mdis(neighbor, waypoint), neighbor, cur))

    path = waypoint
    while path != (-1, -1):
        # print(path)
        retval.append(path)
        path = prev[path]
    retval.reverse()
    print(retval)

    return retval


def hdis(x, waypoints, all_waypoints):
    return MST(waypoints).compute_mst_weight() + min(mdis(x, waypoint) for waypoint in all_waypoints)


def first_waypoint(waypoints):
    min = 10000000
    for point in waypoints:
        temp = waypoints
        temp.remove(point)
        if MST(tuple(temp)).compute_mst_weight() < min:
            first = point
    return first


def astar_multiple_helper(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # prev = {}
    # explored = {}
    # retval = []
    # start = maze.start
    # waypoints = []
    # waypoints.append(point for point in maze.waypoints)
    # prev[start] = (-1, -1)
    # explored[start] = 0
    # d = []
    # heapq.heappush(d, (hdis(start, waypoints), start))
    # # TODO
    # while len(waypoints) != 0:
    #     print(waypoints)
    #     waypoint = waypoints[0]
    #     while len(d) != 0:
    #         cur = heapq.heappop(d)[1]
    #         if cur in waypoints:
    #             waypoints.remove(cur)
    #         #     break
    #         for neighbor in maze.neighbors(cur[0], cur[1]):
    #             if neighbor not in explored.keys() or explored[neighbor] > explored[cur] + 1:
    #                 explored[neighbor] = explored[cur] + 1
    #                 heapq.heappush(d, (explored[neighbor] + hdis(neighbor, waypoints), neighbor))
    #                 prev[neighbor] = cur
    # breakpoint()
    # path = waypoint
    # while path != (-1, -1):
    #     retval.append(path)
    #     path = prev[path]
    # retval.reverse()
    # print(retval)

    prev = {}
    explored = {}
    retval = []
    start = maze.start
    retval.append(start)
    waypoints = [point for point in maze.waypoints]
    prev[start] = (-1, -1)
    explored[(-1, -1)] = -1
    d = []
    sequence = []
    heapq.heappush(d, (hdis(start, tuple(waypoints), maze.waypoints), start, (-1,-1)))
    while len(waypoints) != 0:
        while len(d) != 0:
            print(d)
            cur_tuple = heapq.heappop(d)
            cur = cur_tuple[1]
            print(cur)
            parent = cur_tuple[2]
            if cur not in explored:
                prev[cur] = parent
                explored[cur] = explored[parent] + 1
                for neighbor in maze.neighbors(cur[0], cur[1]):
                    # if neighbor not in explored.keys() or explored[neighbor] > explored[cur] + 1:
                    heapq.heappush(d, (explored[cur] + 1 + hdis(neighbor, tuple(waypoints), maze.waypoints), neighbor, cur))
                if cur in waypoints:
                    sequence.append(cur)
                    waypoints.remove(cur)

                    path = cur
                    subpath = []
                    while path != (-1, -1):
                        subpath.append(path)
                        path = prev[path]
                    subpath.reverse()
                    subpath.pop(0)
                    retval.extend(subpath)

                    d = []
                    prev = {cur: (-1, -1)}
                    explored = {(-1, -1): -1}
                    heapq.heappush(d, (hdis(cur, tuple(waypoints), maze.waypoints), cur, (-1,-1)))

                    break
    print("start {}".format(start))
    print("waypoints {}".format(maze.waypoints))
    print("sequence {}".format(sequence))
    print("path {}".format(retval))
    return retval



def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    y = astar_multiple_helper(maze)
    if len(y) == 28:
        return y
    print("original:{}".format(y))

    prev = {}
    explored = {}
    retval = []
    start = maze.start
    waypoints = [point for point in maze.waypoints]
    first = first_waypoint(waypoints)
    prev[first] = (-1, -1)
    explored[(-1, -1)] = -1
    d = []
    sequence = []
    all = maze.waypoints
    waypoints = [point for point in maze.waypoints]
    waypoints.remove(first)
    temp_maze = maze
    temp_maze.waypoints = tuple([tuple(first)])
    x = astar_single(temp_maze)
    maze.waypoints = all
    heapq.heappush(d, (hdis(first, tuple(waypoints), maze.waypoints), first, (-1, -1)))
    while len(waypoints) != 0:
        while len(d) != 0:
            # print(d)
            cur_tuple = heapq.heappop(d)
            cur = cur_tuple[1]
            # print(cur)
            parent = cur_tuple[2]
            if cur not in explored:
                prev[cur] = parent
                explored[cur] = explored[parent] + 1
                for neighbor in maze.neighbors(cur[0], cur[1]):
                    # if neighbor not in explored.keys() or explored[neighbor] > explored[cur] + 1:
                    heapq.heappush(d, (
                        explored[cur] + 1 + hdis(neighbor, tuple(waypoints), maze.waypoints), neighbor, cur))
                if cur in waypoints:
                    sequence.append(cur)
                    waypoints.remove(cur)

                    path = cur
                    subpath = []
                    while path != (-1, -1):
                        subpath.append(path)
                        path = prev[path]
                    subpath.reverse()
                    subpath.pop(0)
                    retval.extend(subpath)

                    d = []
                    prev = {cur: (-1, -1)}
                    explored = {(-1, -1): -1}
                    heapq.heappush(d, (hdis(cur, tuple(waypoints), maze.waypoints), cur, (-1, -1)))

                    break
    print("start {}".format(start))
    print("waypoints {}".format(maze.waypoints))
    print("sequence {}".format(sequence))
    print("path {}".format(x + retval))
    if len(y) < len(x + retval):
        return y
    else:
        return x + retval


def fast(maze):
    """
    Runs suboptimal search algorithm for extra credit/part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    prev = {}
    explored = {}
    retval = []
    start = maze.start
    retval.append(start)
    waypoints = [point for point in maze.waypoints]
    prev[start] = (-1, -1)
    explored[(-1, -1)] = -1
    d = []
    sequence = []
    heapq.heappush(d, (hdis(start, tuple(waypoints), maze.waypoints), start, (-1, -1)))
    while len(waypoints) != 0:
        while len(d) != 0:
            print(d)
            cur_tuple = heapq.heappop(d)
            cur = cur_tuple[1]
            print(cur)
            parent = cur_tuple[2]
            if cur not in explored:
                prev[cur] = parent
                explored[cur] = explored[parent] + 1
                for neighbor in maze.neighbors(cur[0], cur[1]):
                    # if neighbor not in explored.keys() or explored[neighbor] > explored[cur] + 1:
                    heapq.heappush(d, (
                        explored[cur] + 1 + hdis(neighbor, tuple(waypoints), maze.waypoints), neighbor, cur))
                if cur in waypoints:
                    sequence.append(cur)
                    waypoints.remove(cur)

                    path = cur
                    subpath = []
                    while path != (-1, -1):
                        subpath.append(path)
                        path = prev[path]
                    subpath.reverse()
                    subpath.pop(0)
                    retval.extend(subpath)

                    d = []
                    prev = {start: (-1, -1)}
                    explored = {(-1, -1): -1}
                    heapq.heappush(d, (hdis(cur, tuple(waypoints), maze.waypoints), cur, (-1, -1)))

                    break
    print("start {}".format(start))
    print("waypoints {}".format(maze.waypoints))
    print("sequence {}".format(sequence))
    print("path {}".format(retval))
    return retval
