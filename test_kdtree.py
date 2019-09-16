import unittest
import numpy as np
from numpy import linalg
import operator

from kdtree import Node, Bound, PriorityStack, KDTree

class TestBound(unittest.TestCase):
    def setUp(self):
        dim = 3
        bound = [
            [-np.inf, np.inf],
            [2, np.inf],
            [-np.inf, -3]
        ]
        
        self.bound = Bound(dim, init=bound)
        self.test_point = [0, 0, 0]
        self.test_dist = 1

    def test_dist(self):
        self.assertEqual(
            self.bound.dist(self.test_point),
            linalg.norm([0, 2, 3])
        )

    def test_split(self):
        left, right = self.bound.split([4, 10, -10], 0)

        self.assertEqual(
            left.dist(self.test_point),
            linalg.norm([0, 2, 3])
        )
        self.assertEqual(
            right.dist(self.test_point),
            linalg.norm([4, 2, 3])
        )        

        left, right = self.bound.split([4, 10, -10], 1)

        self.assertEqual(
            left.dist(self.test_point),
            linalg.norm([0, 2, 3])
        )
        self.assertEqual(
            right.dist(self.test_point),
            linalg.norm([0, 10, 3])
        )        
        
        left, right = self.bound.split([4, 10, -10], 2)
        
        self.assertEqual(
            left.dist(self.test_point),
            linalg.norm([0, 2, 10])
        )
        self.assertEqual(
            right.dist(self.test_point),
            linalg.norm([0, 2, 3])
        )        


class TestPriorityStack(unittest.TestCase):
    def setUp(self):
        self.size = 5
        self.stack = PriorityStack(self.size)
        self.stack.push((1, "apple"))
        self.stack.push((10, "egg"))
        self.stack.push((19, "pc"))
        self.stack.push((33, "number"))
        self.stack.push((7, "aha"))
        self.stack.push((9, "job"))

    def test_pop(self):
        self.assertTupleEqual(self.stack.pop(), (19, "pc"))
        self.assertTupleEqual(self.stack.pop(), (10, "egg"))
        self.assertTupleEqual(self.stack.pop(), (9, "job"))
        self.assertTupleEqual(self.stack.pop(), (7, "aha"))
        self.assertTupleEqual(self.stack.pop(), (1, "apple"))

    def test_full(self):
        self.assertTrue(self.stack.full())
        self.stack.pop()
        self.assertFalse(self.stack.full())

    def test_empty(self):
        self.assertFalse(self.stack.empty())
        for i in range(self.size):
            self.stack.pop()
        self.assertTrue(self.stack.empty())

    def test_top(self):
        self.assertTupleEqual(self.stack.top(), (19, "pc"))
        self.stack.pop()
        self.assertTupleEqual(self.stack.top(), (10, "egg"))

    def test_list(self):
        self.assertListEqual(
            self.stack.list(),
            ["apple", "aha", "job", "egg", "pc"]
        )


class TestKDTree(unittest.TestCase):
    def setUp(self):
        self.dim = 3
        self.count = 20
        points = np.random.randint(low=0, high=20, size=(self.count, self.dim)).tolist()

        self.tree = KDTree(self.dim)
        for p in points:
            self.tree.insert(p)

        np.random.shuffle(points)
        self.points = points

    def test_contain(self):
        for p in self.points:
            self.assertTrue(self.tree.contain(p))

        random_points = np.random.randint(low=30, size=(self.count, self.dim)).tolist()
        for p in random_points:
            self.assertFalse(self.tree.contain(p))

    def test_find_min(self):
        for d in range(self.dim):
            sorted_at_dim = sorted(self.points, key=operator.itemgetter(d))
            min_at_dim = sorted_at_dim[0][d]
            candidate_min_at_dim = list(filter(lambda point: point[d] == min_at_dim, sorted_at_dim))
            
            self.assertIn(self.tree.find_min(d, 0, self.tree.root), candidate_min_at_dim)

    def test_delete(self):
        for p in self.points:
            self.tree.delete(p)
            self.assertFalse(self.tree.contain(p))

    def test_nearest_neighbor(self):
        for p in self.points:
            self.assertListEqual(self.tree.find_nearest(p), p)
        
        random_points = np.random.randint(low=0, high=20, size=(20, self.dim))
        for p in random_points:
            nearest_point = self.tree.find_nearest(p)
            points = np.array(self.points)

            self.assertEqual(
                min(linalg.norm(points - p, axis=1)),
                linalg.norm(nearest_point - p)
            )

    def test_k_nearest_neighbor(self):
        for p in self.points:
            self.assertListEqual(self.tree.find_k_nearest(p, 1), [p])

        random_points = np.random.randint(low=0, high=20, size=(20, self.dim))
        for p in random_points:
            nearest_points = self.tree.find_k_nearest(p, 5)
            nearest_points = np.array(nearest_points)
            points = np.unique(self.points, axis=0)

            n_dists = linalg.norm(nearest_points - p, axis=1)
            dists = np.sort(linalg.norm(points - p, axis=1))[:5]

            self.assertListEqual(n_dists.tolist(), dists.tolist())
