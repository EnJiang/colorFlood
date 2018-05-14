import numpy as np
from copy import deepcopy

class Node(object):
    def __init__(self, game):
        self.game = game
        self.N = 0
        self.N_a = [0 for _ in range(6)]
        self.q_values = [0 for _ in range(6)]

        self.children_hash_list = self.cal_child_node_hash()

    def cal_child_node_hash(self):
        hash_list = dict()
        actions = [i + 1 for i in range(6)]
        for a in actions:
            tmp_game = deepcopy(self.game)
            tmp_game.change(a)
            hash_list[a] = tmp_game.hash_string()

    @property
    def ucb_list(self):
        pass

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
        while self.all_children_in_tree(root):
            chian.append(repr(root))
            # now select by ucb
            root = self.next_search_down_node(root)
        
        # now the root node is not full
        child = self.select_and_create_nonexist_child(root)
        p = self.predict_value(child)

        self.back_prop(chian, p)

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
                child = Node(tmp_game)
                self.tree[hash(child)] = child
        return child

    def next_search_down_node(self, root):
        ucb_list = root.ucb_list
        child_index = np.argmax(ucb_list)
        child_hash = root.children_hash_list[child_index]
        return self.tree[child_hash]

    @property
    def pi(self):
        pass
