import torch
from alpha_go_utils.mcts import *
from env import Env
import numpy as np
from copy import deepcopy
from greedy import greedy

import concurrent.futures

from collections import namedtuple

Report = namedtuple("Report", "step path greedy_step")

SEARCH_TIME = 2000

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


def old_main():
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
            t.run(time=1000)

            epoch_pi.append(t.pi)
            epoch_a.append(a)
            a *= 0.8
            
            action_index = np.argmax(t.pi)

            print(t.pi)
            # print(e.game)
            # print(action_index)
            # print()
            obs, reward, done, _ = e.step(action_index)
            eopch_obs.append(obs)

        # now that the game is done, we had a terminal obs
        # add its pi, and another a, and reverse epoch_a
        epoch_pi.append([0 for _ in range(6)])
        epoch_a.append(a)
        epoch_a.reverse()

if __name__ == "__main__":
    model = torch.load("light_trained_pre_cnn.pkl").cuda()
    model.eval()
    model.share_memory()

    mcts_step = 0
    greedy_step = 0

    for i in range(100):
        _, _, report = generate_epoch_training_data(model)

        mcts_step += report.step
        greedy_step += report.greedy_step

        print('''
        mcts   :  %.4f
        greedy :  %.4f
        ''' % (
            mcts_step / (i + 1), greedy_step / (i + 1)
            )
        )
            
