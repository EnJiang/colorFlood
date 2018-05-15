import numpy as np
from copy import deepcopy
from math import sqrt

class Node(object):
    def __init__(self, game, policy, c=1):
        self.game = game
        self.N = 0
        self.N_a_list = [0 for _ in range(6)]
        self.q_a_list = [0 for _ in range(6)]
        self.p_a_list = policy
        self.c = 1
        self.children_hash_list = self.cal_child_node_hash()

    def cal_child_node_hash(self):
        hash_list = dict()
        actions = [i + 1 for i in range(6)]
        for a in actions:
            tmp_game = deepcopy(self.game)
            tmp_game.change(a)
            hash_list[a] = tmp_game.hash_string()

    def ucb_list(self, tree):
        q_a_list = self.q_a_list
        p_a_list = self.p_a_list
        N_a_list = self.N_a_list
        return [
            self.cal_ucb(q_a, p_a, N_a)
            for q_a, p_a, N_a in zip(
                q_a_list, p_a_list, N_a_list
            )
        ]

    def cal_ucb(self, q_a, p_a, N_a):
        return q_a + self.c * p_a * sqrt(self.N) / (1 + N_a)

    def __repr__(self):
        return self.game.hash_string()

    def __hash__(self):
        return self.__repr__()

class MCTS(object):
    def __init__(self, root_node, net):
        self.net = net
        self.tree = dict()
        self.tree[hash(root_node)] = root_node

    def run(self, root):
        if self.all_children_in_tree(root):
            self.search_down(root)
        else:
            self.select_and_create_nonexist_child(root)

    def search_down(self, root):
        chian = []
        next_root = None
        while self.all_children_in_tree(root):
            # now select by ucb
            next_root, child_index = self.next_search_down_node(root)
            chian.append((hash(root), child_index))
            root = next_root
        
        # now the root node is not full
        child = self.select_and_create_nonexist_child(root)
        v = self.predict_value(child.game)

        self.back_prop(chian, v)

    def all_children_in_tree(self, root):
        return all(
            map(
                lambda x: x in self.tree.keys(),
                root.children_hash_list
            )
        )

    def select_and_create_nonexist_child(self, root):
        if self.all_children_in_tree(root):
            raise ValueError("All children in tree!")
        for i in range(6):
            if root.children_hash_list[i] not in self.tree.keys():
                action = i + 1
                tmp_game = deepcopy(root.game)
                tmp_game.change(action)
                child = Node(tmp_game, self.predict_policy(tmp_game))
                self.tree[hash(child)] = child
        return child

    def next_search_down_node(self, root):
        ucb_list = root.ucb_list(self.tree)
        child_index = np.argmax(ucb_list)
        child_hash = root.children_hash_list[child_index]
        return self.tree[child_hash], child_index

    def predict_value(self, game):
        pass

    def predict_policy(self, game):
        pass

    @property
    def pi(self):
        pass
