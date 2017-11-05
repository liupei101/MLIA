# coding=utf-8
import numpy as numpy

# sized Priority Queue
class BoundedPriorityQueue:
    def __init__(self, k):
        self.heap=[]
        self.k = k

    def items(self):
        return self.heap

    def parent(self,index):
        return int(index / 2)

    def left_child(self, index):
        return 2*index + 1

    def right_index(self,index):
        return 2*index + 2

    def _dist(self, index):
        return self.heap[index][3]

    def max_heapify(self, index):
        left_index = self.left_child(index)
        right_index = self.right_index(index)

        largest = index
        if left_index <len(self.heap) and self._dist(left_index) >self._dist(index):
            largest = left_index
        if right_index <len(self.heap) and self._dist(right_index) > self._dist(largest):
            largest = right_index
        if largest != index :
            self.heap[index], self.heap[largest] =  self.heap[largest], self.heap[index]
            self.max_heapify(largest)

    def propagate_up(self,index):
        while index != 0 and self._dist(self.parent(index)) < self._dist(index):
            self.heap[index], self.heap[self.parent(index)] = self.heap[self.parent(index)],self.heap[index]
            index = self.parent(index)

    def add(self, obj):
        size = self.size()
        if size == self.k:
            max_elem = self.max()
            if obj[1] < max_elem:
                self.extract_max()
                self.heap_append(obj)
        else:
            self.heap_append(obj)

    def heap_append(self, obj):
        self.heap.append(obj)
        self.propagate_up(self.size()-1)

    def size(self):
        return len(self.heap)

    def max(self):
        return self.heap[0][4]

    def extract_max(self):
        max = self.heap[0]
        data = self.heap.pop()
        if len(self.heap)>0:
            self.heap[0]=data
            self.max_heapify(0)
        return max 

# Class - Node
class Node(object):
	""" 
	Initialize a Node
	"""
	def __init__(self, data = None, left = None, right = None):
		self.data = data
		self.left = left
		self.right = right

# Class - KD-Tree Node
class KDNode(Node):
	"""
	Initialize a Node of KD-Tree
	"""
	def __init__(self, data = None, left = None, right = None, axis = None, sel_axis = None, dimensions = None):
		super(KDNode, self).__init__(data, left, right)
		self.axis = axis
		self.sel_axis = sel_axis
		self.dimensions = dimensions

# create Tree
def create(point_list = None, dimensions = None, axis = 0, sel_axis = None):
	if not point_list and not dimensions:
		raise ValueError("either point_list or dimensions should be provided.")
	elif point_list:
		dimensions = check_dimensionality(point_list, dimensions)


	if not point_list:
		return KDNode(sel_axis = sel_axis, axis = axis, dimensions = dimensions)

	# split Node
	point_list = list(point_list)
	point_list.sort(key = lambda point:point[axis])
	M = len(point_list) // 2
	Val = point_list[M]

	left = create(point_list[:M], dimensions, (axis + 1) % dimensions)
	right = create(point_list[M+1:], dimensions, (axis + 1) % dimensions)
	return KDNode(val, left, right, axis = axis, sel_axis = sel_axis, dimensions = dimensions)

def check_dimensionality(point_list,dimensions=None):
    dimensions = dimensions or len(point_list[0])
    for p in point_list:
        if len(p) != dimensions:
            raise ValueError('All Points in the point_list must have the same dimensionality')
    return dimensions

def _search_node(self, point, k, results, get_dist):
        if not self:
            return
        nodeDist = get_dist(self)

        results.add((self,nodeDist))

        split_plane = self.data[self.axis]
        plane_dist = point[self.axis] - split_plane
        plane_dist2 = plane_dist ** 2

        if point[self.axis] < split_plane:
            if self.left is not None:
                self.left._search_node(point, k, results, get_dist)
        else:
            if self.right is not None:
                self.right._search_node(point, k, results, get_dist)

        # Key Point
        if plane_dist2 < results.max() or results.size() < k:
            if point[self.axis] < self.data[self.axis]:
                if self.right is not None:
                    self.right._search_node(point, k, results, get_dist)
            else:
                if self.left is not None:
                    self.left._search_node(point, k, results, get_dist)

# search k nearest neighbers
def search_knn(self, point, k, dist = None):
        if dist is None:
            get_dist = lambda n : n.dist(point)
        else:
            gen_dist = lambda n : dist(n.data, point)

        results = BoundedPriorityQueue(k)
        self._search_node(point, k, results, get_dist)

        BY_VALUE = lambda kv: kv[1]
        return sorted(results.items(), key = BY_VALUE)