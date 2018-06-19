import torch
from torch.utils.data import DataLoader

import numpy as np
from copy import deepcopy

from env import Env
from greedy import greedy

from torch.utils.data import DataLoader
from alpha_go_utils.data import MyDataset

if __name__ == "__main__":
    model = torch.load("pre_cnn.pkl").cuda()
    # vxs = np.load(open("vxs.npy", "rb"))
    # vys = np.load(open("vys.npy", "rb"))
    # vset = MyDataset(vxs, vys)
    # vdataloader = DataLoader(vset, batch_size=8, shuffle=False)

    model.eval()
    # loss = 0
    # for batch_idx, sample in enumerate(vdataloader):
    #     # print(type(sample))
    #     inputs, targets = sample[0], sample[1]
    #     inputs, targets = inputs.cuda(), targets.cuda()
    #     outputs = model(inputs)
    #     print(outputs)
    #     print(targets)

    count_network = 0
    count_greedy = 0
    test_rount = 300

    e = Env(size=6)

    for _ in range(test_rount):
        print(_)
        done = False
        obs = e.reset()
        g_e = deepcopy(e)

        obs = np.reshape(obs, (1, 4, 6, 6))
        obs = torch.FloatTensor(obs).cuda()
        while not done:
            output = model(obs)
            output = output.cpu().data.numpy()[0]
            pi = output[: 6]
            a = output[-1]

            # action_greedy = greedy(e.game, 1)[0] - 1
            action_pi = np.argmax(pi)

            # print(e.game)
            # print(action_greedy, action_pi, a)
            # print(pi)

            next_obs, reward, done, _ = e.step(action_pi)
            obs = next_obs
            obs = np.reshape(obs, (1, 4, 6, 6))
            obs = torch.FloatTensor(obs).cuda()

        done = False
        while not done:
            action_index = greedy(g_e.game, 1)[0] - 1
            next_obs, reward, done, _ = g_e.step(action_index)
        greedy_step = g_e.game.step

        # print(e.game.step, g_e.game.step)
        # print(e.game.allStep, g_e.game.allStep)
        # print()
        count_network += e.game.step
        count_greedy += g_e.game.step

    print(
        '''
        cnn   : avg %.4f step
        greedy: avg %.4f step
        ''' % 
        (count_network / test_rount, count_greedy / test_rount)
    )
