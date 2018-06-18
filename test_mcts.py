from alpha_go_utils.mcts import *
from env import Env
import numpy as np
from copy import deepcopy
from greedy import greedy

if __name__ == "__main__":
    # test here
    # model = torch.load("pre_cnn.pkl").cuda()
    e = Env(size=6)
    done = False
    e.reset()
    for _ in range(100):
        done = False
        e.reset()
        g_e = deepcopy(e)

        while not done:
            root_node = init_node(e, use_nn=False)
            t = MCTS(root_node)
            t.run(time=5000)
            # print(t.pi)
            action_index = np.argmax(t.pi)
            # print(e.game)
            # print(action_index)
            # print()
            next_obs, reward, done, _ = e.step(action_index)

        done = False
        while not done:
            action_index = greedy(g_e.game, 1)[0] - 1
            next_obs, reward, done, _ = g_e.step(action_index)
        greedy_step = g_e.game.step

        print(e.game.step, g_e.game.step)
        print(e.game.allStep, g_e.game.allStep)
        print()