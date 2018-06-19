import torch
from alpha_go_utils.mcts import *
from env import Env
import numpy as np
from copy import deepcopy
from greedy import greedy

if __name__ == "__main__":
    model = torch.load("light_trained_pre_cnn.pkl").cuda()
    model.eval()

    obs_memory = []
    act_memory = []

    e = Env(size=6)
    done = False
    e.reset()
    for _ in range(100):
        done = False

        obs = e.reset()
        g_e = deepcopy(e)

        eopch_obs = [obs]
        epoch_pi = []
        epoch_a = []

        a = 1

        while not done:
            root_node = init_node(e, use_nn=True, model=model)
            t = MCTS(root_node, use_nn=True, net=model)
            t.run(time=3000)

            epoch_pi.append(t.pi)
            epoch_a.append(a)
            a *= 0.8
            
            action_index = np.argmax(t.pi)

            print(t.pi)
            print(e.game)
            # print(action_index)
            # print()
            obs, reward, done, _ = e.step(action_index)
            eopch_obs.append(obs)

        # now that the game is done, we had a terminal obs
        # add its pi, and another a, and reverse epoch_a
        epoch_pi.append([0 for _ in range(6)])
        epoch_a.append(a)
        epoch_a.reverse()

        print(epoch_a)
        print(e.game.step, len(epoch_pi), len(eopch_obs), len(epoch_a))
        # exit()

        done = False
        while not done:
            action_index = greedy(g_e.game, 1)[0] - 1
            next_obs, reward, done, _ = g_e.step(action_index)
        greedy_step = g_e.game.step

        print(e.game.step, g_e.game.step)
        print(e.game.allStep, g_e.game.allStep)
        print()
