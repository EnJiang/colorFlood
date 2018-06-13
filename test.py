import torch
from torch.utils.data import DataLoader

import numpy as np

from env import Env
from greedy import greedy

from torch.utils.data import DataLoader
from alpha_go_utils.data import MyDataset

if __name__ == "__main__":
    model = torch.load("pre_cnn.pkl").cuda()
    vxs = np.load(open("vxs.npy", "rb"))
    vys = np.load(open("vys.npy", "rb"))
    vset = MyDataset(vxs, vys)
    vdataloader = DataLoader(vset, batch_size=8, shuffle=False)

    model.eval()
    # loss = 0
    # for batch_idx, sample in enumerate(vdataloader):
    #     # print(type(sample))
    #     inputs, targets = sample[0], sample[1]
    #     inputs, targets = inputs.cuda(), targets.cuda()
    #     outputs = model(inputs)
    #     print(outputs)
    #     print(targets)


    e = Env(size=6)
    done = False
    obs = e.reset()
    obs = np.reshape(obs, (1, 4, 6, 6))
    obs = torch.FloatTensor(obs).cuda()
    while not done:
        output = model(obs)
        output = output.cpu().data.numpy()[0]
        pi = output[: 6]
        a = output[-1]

        action_greedy = greedy(e.game, 1)[0] - 1
        action_pi = np.argmax(pi)

        print(e.game)
        print(action_greedy, action_pi, a)
        print(pi)

        next_obs, reward, done, _ = e.step(action_greedy)
        obs = next_obs
        obs = np.reshape(obs, (1, 4, 6, 6))
        obs = torch.FloatTensor(obs).cuda()
    print("\n\n\n\n\n\n")
