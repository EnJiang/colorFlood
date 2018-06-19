import numpy as np
from copy import deepcopy
from math import sqrt, inf
import torch
from tqdm import tqdm
import random

class Node(object):
    def __init__(self, env, policy, c=1e-3):
        self.env = env
        self.N = 0
        self.N_a_list = [0 for _ in range(6)]
        self.q_a_list = [0 for _ in range(6)]
        self.p_a_list = policy
        self.c = c
        self.children_hash_list = self.cal_child_node_hash()

        self._ucb_list_cache = None
        self._ucb_N_cache = -1

    def cal_child_node_hash(self):
        hash_list = []
        action_index_list = [i for i in range(6)]
        for action_index in action_index_list:
            tmp_env = deepcopy(self.env)
            tmp_env.step(action_index)
            hash_list.append(tmp_env.game.hash_string())
        return hash_list

    def ucb_list(self):
        q_a_list = self.q_a_list
        p_a_list = self.p_a_list
        N_a_list = self.N_a_list
        ucb_list =  [
            self.cal_ucb(q_a, p_a, N_a)
            for q_a, p_a, N_a in zip(
                q_a_list, p_a_list, N_a_list
            )
        ]
        return ucb_list
        # return self._ucb_list(self.N)

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
        return q_a / (N_a + 1) + self.c * p_a * sqrt(self.N) / (1 + N_a)

    @property
    def is_over(self):
        return self.env.game.isOver()

    def __repr__(self):
        return self.env.game.hash_string()

    def __hash__(self):
        raise NotImplementedError()

class MCTS(object):
    def __init__(self, root_node, net=None, use_nn=False, size=6):
        assert not (net is None and use_nn == True)
        self.use_nn = use_nn
        self.net = net
        self.tree = dict()
        self.tree[repr(root_node)] = root_node
        self.root_node = root_node
        self.size = size

    def run(self, time=1):
        for _ in tqdm(range(time)):
            self.search_down(self.root_node)

    def search_down(self, root):
        chain = []
        next_root = None

        while self.all_children_in_tree(root) and not root.is_over:
            '''now select by ucb'''
            # print("current root %r" % root)
            next_root, child_index = self.next_search_down_node(root)
            '''child_index is actually an action index'''
            # print("all child in tree! next: %r" % next_root)
            chain.append((repr(root), child_index))
            root = next_root
        
        '''now the root node is not full exlpored,
           let's select a unseen child'''
        if not root.is_over:
            child, action_index = self.select_and_create_nonexist_child(root)
            chain.append((repr(root), action_index))
            v = self.predict_value(child.env)
        else:
            v = 1.0

        # print([one[1] for one in chain], v)
        self.back_prop(chain, v)

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
        for action_index in range(self.size):
            if (root.children_hash_list[action_index] not in self.tree.keys()):
                tmp_env = deepcopy(root.env)
                tmp_env.step(action_index)
                child = Node(tmp_env, self.predict_policy(tmp_env))
                self.tree[repr(child)] = child
                break # explore only one child
        return child, action_index

    def next_search_down_node(self, root):
        ucb_list = root.ucb_list()
        root_area = root.env.game.targetArea()

        # lower layer of the tree mush has larger
        # target area
        for i, hash_string in enumerate(root.children_hash_list):
            child = self.tree[hash_string]
            child_area = child.env.game.targetArea()
            if child_area <= root_area:
                ucb_list[i] = -inf

        child_index = np.argmax(ucb_list)
        child_hash = root.children_hash_list[child_index]

        return self.tree[child_hash], child_index

    def network_forward(self, env):
        obs = env.last_obs
        obs = np.reshape(obs, (1, ) + obs.shape)
        obs = torch.FloatTensor(obs).cuda()
        output = self.net(obs)
        output = output.cpu().data.numpy()[0]
        return output

    def predict_policy(self, env):
        if self.use_nn:
            output = self.network_forward(env)
            return output[: self.size]
        else:
            return [1 for _ in range(self.size)]

    def predict_value(self, env):
        if self.use_nn:
            output = self.network_forward(env)
            return output[-1]
        return env.game.targetArea() / (self.size ** 2)

    def back_prop(self, chain, v):
        for hash_string, action_index in chain:
            node = self.tree[hash_string]
            node.update(action_index, v)

    @property
    def pi(self):
        last_action_index = self.root_node.env.last_action_index
        q_a_list = self.root_node.q_a_list
        N_a_list = self.root_node.N_a_list
        # result = [q / N for q, N in zip(q_a_list, N_a_list)]
        result = N_a_list
        result[last_action_index] = 0
        return result

def init_node(env, model=None, use_nn=False):
    assert not (model is None and use_nn == True)
    if use_nn:
        model.eval()
        obs = env.last_obs
        obs = np.reshape(obs, (1, 4, 6, 6))
        obs = torch.FloatTensor(obs).cuda()
        output = model(obs)
        output = output.cpu().data.numpy()[0]
        pi = list(output[: 6])
        model.train()
    else:
        pi = [1 for _ in range(6)]

    node = Node(env, pi)
    return node