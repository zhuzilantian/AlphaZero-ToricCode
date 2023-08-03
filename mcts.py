import numpy as np
from typing import List, Union


class Node:
    def __init__(self, parent=None, observation=None):
        self.observation = observation
        self.n = 0
        self.q_accumulation = 0
        self.parent: Node = parent
        self.children: List[Node] = []
        self.best_child_score = float('-inf')
        self.best_child_index = 0

    def new_result(self, result: Union[int, float]):
        self.n += 1
        self.q_accumulation += result

    def q_average(self):
        return (self.q_accumulation / self.n
                if self.n > 0
                else 0)

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def evaluation(self, c: Union[int, float]):
        raise NotImplementedError


class NodeUCT(Node):
    def __init__(self, parent=None, observation=None):
        super().__init__(parent, observation)

    def evaluation(self, c=1):
        return (self.q_accumulation / self.n + c * np.sqrt(np.log(self.parent.n) / self.n)
                if self.n > 0
                else float('inf'))


class NodePUCT(Node):
    def __init__(self, p=1, parent=None, observation=None):
        super().__init__(parent, observation)
        self.p = p

    def evaluation(self, c=1):
        return self.q_average() + c * self.p * np.sqrt(self.parent.n) / (1 + self.n)


def selection(node: Node, c=1):
    if node is None:
        return None, None

    path = []
    while not node.is_leaf():
        scores = [n.evaluation(c) for n in node.children]
        index = scores.index(max(scores))
        node = node.children[index]
        path.append(index)
    return node, path


def expansion(node: Node, n: int, p=None):
    if node is None:
        return

    for i in range(n):
        if p is not None and len(p) >= n:
            child = NodePUCT(p=p[i], parent=node)
        else:
            child = NodeUCT(parent=node)
        node.children.append(child)


def backpropagation(node: Node, result):
    if node is None:
        return

    if not node.is_root():
        backpropagation(node.parent, result)
    node.new_result(result)
