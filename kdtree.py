"""kd tree"""

import numpy as np
import copy

class Node():
    """kd tree node"""
    def __init__(self, point):
        self.point = point
        self.left = None
        self.right = None


class Bound():
    """n dimension space bound"""
    def __init__(self, dim, init=None):
        self.dim = dim
        if init is None:
            self.bound = []
            for d in range(dim):
                self.bound.append([-np.inf, np.inf])
        else:
            self.bound = init

    def dist(self, point):
        """
        the min dist between the point and BB itself
        """
        mins = []
        for d in range(self.dim):
            bound_at_dim = self.bound[d]
            point_at_dim = point[d]

            if point_at_dim < bound_at_dim[0]:
                mins.append(bound_at_dim[0] - point_at_dim)
            elif point_at_dim > bound_at_dim[1]:
                mins.append(point_at_dim - bound_at_dim[1])
            else:
                mins.append(0)

        return np.linalg.norm(mins)

    def split(self, point, dim):
        """
        in dimension `dim`, split bound into left and right with point `point`
        """
        bound_at_dim = self.bound[dim]
        point_at_dim = point[dim]

        if point_at_dim < bound_at_dim[0] or point_at_dim > bound_at_dim[1]:
            raise RuntimeError("can't split bound with point at dimension")

        left_bound = copy.deepcopy(self.bound)
        right_bound = copy.deepcopy(self.bound)

        left_bound[dim][1] = point_at_dim
        right_bound[dim][0] = point_at_dim

        return Bound(self.dim, init=left_bound), Bound(self.dim, init=right_bound)


class PriorityStack():
    """
    priority stack with fixed size.

    only elements with the first `size`-th priority stay in the stack.
    by default, the less the number is, the higher the priority is.
    """
    def __init__(self, size):
        self.stack = []
        self.size = size

    def push(self, ele):
        """
        push element into stack

        the priority depends on the first part of element.
        e.g.
        - element 2 with priority 2
        - element (4, "hello") with priority 4
        """
        self.stack.append(ele)
        self.stack.sort(key=lambda e: list(e)[0])
        if len(self.stack) > self.size:
            self.stack.pop()

    def pop(self):
        """pop the top element"""
        return self.stack.pop()

    def full(self):
        """if stack is full"""
        return len(self.stack) == self.size

    def empty(self):
        """if stack is empty"""
        return len(self.stack) == 0

    def top(self):
        """query the top element without popping it"""
        if self.empty():
            return None
        
        return self.stack[-1]

    def list(self):
        """fetch the stack elements as a list with the priority order"""
        return list(map(lambda t: t[1], self.stack))


class KDTree():
    """kd tree"""
    def __init__(self, dim):
        self.dim = dim
        self.root = None

    def insert(self, point):
        """insert `point` into kd tree"""
        def _insert(point, cur_dim, root):
            """at current dimension `dim`, insert `point` into kd tree with root `root`"""
            if root is None:
                return Node(point)

            if point == root.point:
                print("Warning: duplicate point occurs")
                print(point)
                return root
            
            cur_dim = cur_dim % self.dim
            next_dim = cur_dim + 1

            if point[cur_dim] < root.point[cur_dim]:
                root.left = _insert(point, next_dim, root.left)
            else:
                root.right = _insert(point, next_dim, root.right)

            return root

        self.root = _insert(point, 0, self.root)

    def contain(self, point):
        """if `point` in kd tree"""
        def _contain(point, cur_dim, root):
            """at current dimension `dim`, if `point` in kd tree with root `root`"""
            if root is None:
                return False

            if point == root.point:
                return True

            cur_dim = cur_dim % self.dim
            next_dim = cur_dim + 1

            if point[cur_dim] < root.point[cur_dim]:
                return _contain(point, next_dim, root.left)
            else:
                return _contain(point, next_dim, root.right)

        return _contain(point, 0, self.root)

    def find_min(self, dim, cur_dim, root):
        """at dimension `cur_dim`, find the min point in dimension `dim` in kd tree with root `root`"""
        if root is None:
            return None

        min_at_dim = root.point
        
        def _find_min(dim, cur_dim, root):
            nonlocal min_at_dim
            
            if root is None:
                return

            cur_dim = cur_dim % self.dim
            next_dim = cur_dim + 1

            if root.point[dim] < min_at_dim[dim]:
                min_at_dim = root.point
            
            if dim == cur_dim:
                _find_min(dim, next_dim, root.left)
            else:
                _find_min(dim, next_dim, root.left)                
                _find_min(dim, next_dim, root.right)

        _find_min(dim, cur_dim, root)
        return min_at_dim
                
    def delete(self, point):
        """delete `point` from kd tree"""
        def _delete(point, cur_dim, root):
            if root is None:
                return None

            cur_dim = cur_dim % self.dim
            next_dim = cur_dim + 1
            
            if point == root.point:
                if root.left is None and root.right is None:
                    root = None
                elif root.right is None:
                    root.point = self.find_min(cur_dim, next_dim, root.left)
                    root.right = _delete(root.point, next_dim, root.left)
                    root.left = None
                else: # root.right is not None
                    root.point = self.find_min(cur_dim, next_dim, root.right)
                    root.right = _delete(root.point, next_dim, root.right)
            else: # point != root.point
                if point[cur_dim] < root.point[cur_dim]:
                    root.left = _delete(point, next_dim, root.left)
                else:
                    root.right = _delete(point, next_dim, root.right)
                    
            return root
                    
        self.root = _delete(point, 0, self.root)

    def _dist(self, a, b):
        """euclidean distance"""
        from scipy.spatial import distance
        return distance.euclidean(a, b)

    def find_nearest(self, point):
        """find the nearest point to `point` in kd tree"""
        if self.root is None:
            return None

        # n_ means nearest
        n_point = None
        n_dist = np.inf
        bound = Bound(self.dim)

        def _find_nearest(point, root, cur_dim, bound):
            nonlocal n_dist, n_point
            
            if root is None:
                return

            if n_dist <= bound.dist(point):
                return
            
            d = self._dist(root.point, point)
            if d < n_dist:
                n_point = root.point
                n_dist = d

            cur_dim = cur_dim % self.dim
            next_dim = cur_dim + 1

            left_bound, right_bound = bound.split(root.point, cur_dim)

            if point[cur_dim] < root.point[cur_dim]:
                _find_nearest(point, root.left, next_dim, left_bound)
                _find_nearest(point, root.right, next_dim, right_bound)
            else:
                _find_nearest(point, root.right, next_dim, right_bound)
                _find_nearest(point, root.left, next_dim, left_bound)
                
        _find_nearest(point, self.root, 0, bound)
        return n_point

    def find_k_nearest(self, point, k):
        """find `k` points nearest to `point` in kd tree"""
        if self.root is None:
            return None
        
        point_stack = PriorityStack(k)
        bound = Bound(self.dim)
        n_dist = np.inf
        
        def _find_k_nearest(point, root, cur_dim, bound):
            nonlocal n_dist
            
            if root is None:
                return

            if point_stack.full() and n_dist <= bound.dist(point):
                return

            d = self._dist(root.point, point)
            point_stack.push((d, root.point))
            n_dist = point_stack.top()[0]

            cur_dim = cur_dim % self.dim
            next_dim = cur_dim + 1

            left_bound, right_bound = bound.split(root.point, cur_dim)

            if point[cur_dim] < root.point[cur_dim]:
                _find_k_nearest(point, root.left, next_dim, left_bound)
                _find_k_nearest(point, root.right, next_dim, right_bound)
            else:
                _find_k_nearest(point, root.right, next_dim, right_bound)
                _find_k_nearest(point, root.left, next_dim, left_bound)

        _find_k_nearest(point, self.root, 0, bound)
        return point_stack.list()
