import torch
import numpy as np

from env import Env

if __name__ == "__main__":
    model = torch.load("pre_cnn.pkl").cuda()

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

        action_pi = np.argmax(pi)

        print(e.game)
        print(action_pi, a)

        next_obs, reward, done, _ = e.step(action_pi)
        obs = next_obs
        obs = np.reshape(obs, (1, 4, 6, 6))
        obs = torch.FloatTensor(obs).cuda()
    print("\n\n\n\n\n\n")
