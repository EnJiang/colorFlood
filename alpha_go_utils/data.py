from env import Env
from greedy import greedy
from time import time
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from tqdm import tqdm


def generate_greedy(file_dir='./', data_num=10000):
    xs = []
    ys = []
    e = Env(size=6)
    for _ in tqdm(range(data_num)):
        obs_chain = []
        action_chain = []

        done = False
        obs = e.reset()
        while not done:
            action = greedy(e.game, 1)[0] - 1
            next_obs, reward, done, _ = e.step(action)
            obs_chain.append(obs)
            action_chain.append(action)
            obs = next_obs
        
        # now obs is a full board, action is the last action
        # use action = -1 to denote no further action needed
        obs_chain.append(obs)
        action_chain.append(-1)

        chain_length = len(obs_chain)
        for i, (obs, action) in enumerate(zip(obs_chain, action_chain)):
            xs.append(obs)

            one_hot_action = [0 for _ in range(6)]
            if action != -1:
                one_hot_action[action] = 1
            p = 0.8 ** (chain_length - 1 - i)
            ys.append(one_hot_action + [p])
    
    now = str(int(time()))
    xs = np.array(xs)
    ys = np.array(ys)
    np.savez_compressed(
        file_dir + '/' + now, 
        xs= xs, ys=ys)
    return now + ".npz"
