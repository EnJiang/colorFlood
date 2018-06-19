import torch
from alpha_go_utils.mcts import *
from alpha_go_utils.network import *
from env import Env
import numpy as np
from copy import deepcopy
from greedy import greedy

from collections import namedtuple

Report = namedtuple("Report", "step path greedy_step")

SEARCH_TIME = 100

def greedy_evluate(env):
    g_e = deepcopy(env)
    done = False
    while not done:
        action_index = greedy(g_e.game, 1)[0] - 1
        _, _, done, _ = g_e.step(action_index)
    greedy_step = g_e.game.step
    return greedy_step

def generate_epoch_training_data(model):
    model.eval()

    e = Env(size=6)
    obs = e.reset()

    greedy_step = greedy_evluate(e)

    eopch_obs = [obs]
    epoch_pi = []
    epoch_a = []
    a = 1

    done = False
    while not done:
        root_node = init_node(e, use_nn=True, model=model)
        t = MCTS(root_node, use_nn=True, net=model)
        t.run(time=SEARCH_TIME)

        print(e.game)
        print(t.pi)
        epoch_pi.append(t.pi)
        epoch_a.append(a)
        a *= 0.8

        action_index = np.argmax(t.pi)

        obs, _, done, _ = e.step(action_index)
        eopch_obs.append(obs)

    # now that the game is done, we had a terminal obs
    # add its pi, and another a, and reverse epoch_a
    epoch_pi.append([0 for _ in range(6)])
    epoch_a.append(a)
    epoch_a.reverse()

    epoch_output = []
    for pi, a in zip(epoch_pi, epoch_a):
        epoch_output.append(
            pi + [a]
        )

    report = Report(e.game.step, e.game.allStep, greedy_step)

    return eopch_obs, epoch_output, report

if __name__ == "__main__":
    model = torch.load("pre_cnn.pkl").cuda()
    # model = ConvNet().cuda()

    e = Env(size=6)
    e.reset()

    done = False
    while not done:
        step_used = e.game.step
        greedy_step = greedy_evluate(e)
        baseline = greedy_step - step_used

        root_node = init_node(e, use_nn=True, model=model)
        t = MCTS(root_node, use_nn=False, net=model)
        t.run(baseline, time=SEARCH_TIME)

        print()
        print(e.game)
        print(t.pi)
        
        action_index = np.argmax(t.pi)
        _, _, done, _ = e.step(action_index)

    print(e.game.step)
