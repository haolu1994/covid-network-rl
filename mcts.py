import numpy as np
import copy
from operator import itemgetter
from random import choice, shuffle


class Node:
    def __init__(self, A, M, feasible_nodes, actions_remaining, parent=None):
        self.A = A
        self.M = M
        self.feasible_nodes = feasible_nodes
        self.actions_remaining = actions_remaining
        self.n_action_visits = {}
        self.Q = {}
        self.parent = parent
        self.children = []

    def isFullyExpanded(self):
        return len(self.actions_remaining) == 0

    def nextNode(self, action):
        p =
        if np.random.binomial(1, p):
            return Node(A, M, )


def mcts(root, expansion_policy, rollout_policy, T, iterations=2000):
    """
    Monte Carlo Tree Search
    - `expansion_policy` should be a function that takes a node and returns a
    list of child nodes
    - `rollout_policy` should be a function that takes a node and returns a
    reward for that node
    """
    # root.children = expansion_policy(root)

    # MCTS
    for _ in tqdm(range(iterations)):
        step = T
        cur_node = root

        # Selection
        while True:
            if cur_node.n_visits > 0 and cur_node.isFullyExpanded():
                action =
                cur_node = cur_node.best_child()
            else:
                break

        if cur_node.n_visits > 0:
            # If selection took us to a terminal node,
            # this seems to be the best path
            if cur_node.is_terminal:
                break

            # Expansion
            s = time()
            cur_node.children = expansion_policy(cur_node)
            print('Expansion took:', time() - s)
            cur_node = cur_node.best_child()

        # Rollout
        s = time()
        reward = rollout_policy(cur_node, max_depth=max_depth)
        print('Rollout took:', time() - s)

        # Update
        cur_node.reward += reward
        cur_node.n_visits += 1
        parent = cur_node.parent
        while parent is not None:
            parent.reward += reward
            parent.n_visits += 1
            parent = parent.parent

    return action


root = Node(A, M, feasible_nodes)
mcts(root, expansion_policy, rollout_policy, iterations=2000, max_depth=200)