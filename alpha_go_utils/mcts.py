import numpy as np
from copy import deepcopy
from math import sqrt
import torch
from functools import lru_cache

class Node(object):
    def __init__(self, game, policy, c=1):
        self.game = game
        self.N = 0
        self.N_a_list = [0 for _ in range(6)]
        self.q_a_list = [0 for _ in range(6)]
        self.p_a_list = policy
        self.c = 1
        self.children_hash_list = self.cal_child_node_hash()

        self._ucb_list_cache = None
        self._ucb_N_cache = -1

    def cal_child_node_hash(self):
        hash_list = dict()
        actions = [i + 1 for i in range(6)]
        for a in actions:
            tmp_game = deepcopy(self.game)
            tmp_game.change(a)
            hash_list[a] = tmp_game.hash_string()

    def ucb_list(self):
        return self._ucb_list(self.N)

    def _ucb_list(self, N):
        # work a cache here, N works like
        # a key
        if (self._ucb_list_cache is None or
            self._ucb_N_cache != N):
            q_a_list = self.q_a_list
            p_a_list = self.p_a_list
            N_a_list = self.N_a_list
            ucb_list =  [
                self.cal_ucb(q_a, p_a, N_a)
                for q_a, p_a, N_a in zip(
                    q_a_list, p_a_list, N_a_list
                )
            ]
            self._ucb_list_cache = ucb_list
            self._ucb_N_cache = N
            return ucb_list
        else:
            return self._ucb_list_cache

    def update(self, action_index, v):
        self.N += 1
        self.N_a_list[action_index] += 1
        self.q_a_list[action_index] += v

    def cal_ucb(self, q_a, p_a, N_a):
        return q_a + self.c * p_a * sqrt(self.N) / (1 + N_a)

    def __repr__(self):
        return self.game.hash_string()

    def __hash__(self):
        return self.__repr__()

class MCTS(object):
    def __init__(self, root_node, net=None, use_nn=False, size=6):
        assert not (net is None and use_nn == True)
        self.use_nn = use_nn
        self.net = net
        self.tree = dict()
        self.tree[hash(root_node)] = root_node
        self.root_node = root_node
        self.size = size

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
            # child_index is actually an action index
            chian.append((hash(root), child_index))
            root = next_root
        
        # now the root node is not full exlpored,
        # let's select a unseen child
        child, action_index = self.select_and_create_nonexist_child(root)
        chian.append((hash(root), action_index))
        v = self.predict_value(child.game)

        self.back_prop(chian, v)

    def all_children_in_tree(self, root):
        keys = self.tree.keys()
        return all(
            map(
                lambda x: x in keys,
                root.children_hash_list
            )
        )

    def select_and_create_nonexist_child(self, root):
        if self.all_children_in_tree(root):
            raise ValueError("All children in tree!")
        for i in range(self.size):
            if root.children_hash_list[i] not in self.tree.keys():
                action = i + 1
                tmp_game = deepcopy(root.game)
                tmp_game.change(action)
                child = Node(tmp_game, self.predict_policy(tmp_game))
                self.tree[hash(child)] = child
                break # explore select only one child
        return child, action - 1

    def next_search_down_node(self, root):
        ucb_list = root.ucb_list()
        child_index = np.argmax(ucb_list)
        child_hash = root.children_hash_list[child_index]
        return self.tree[child_hash], child_index

    def network_forward(self, game):
        obs = game.obs
        obs = np.reshape(obs, (1, ) + obs.shape)
        obs = torch.FloatTensor(obs).cuda()
        output = self.net(obs)
        output = output.cpu().data.numpy()[0]
        return output

    def predict_value(self, game):
        if self.use_nn:
            output = self.network_forward(game)
            return output[: self.size]
        else:
            return [0 for _ in range(self.size)]


    def predict_policy(self, game):
        if self.use_nn:
            output = self.network_forward(game)
            return output[-1]
        return game.targetArea() // (self.size ** 2)

    def back_prop(self, chain, v):
        for node, action_index in chain:
            node.update(action_index, v)

    @property
    def pi(self):
        obs = self.root_node.game.obs
        obs = np.reshape(obs, (1, ) + obs.shape)
        obs = torch.FloatTensor(obs).cuda()
        output = self.net(obs)
        output = output.cpu().data.numpy()[0]
        pi = output[: 6]
        return pi


if __name__ == "__main__":
    # test here
    pass
