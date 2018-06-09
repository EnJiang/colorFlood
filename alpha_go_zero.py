from alpha_go_utils.data import *
from alpha_go_utils.network import ConvNet
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from env import Env

def validate(model):
    e = Env(size=6)
    done = False
    obs = e.reset()
    obs = np.reshape(obs, (1, 4, 6, 6))
    obs = torch.FloatTensor(obs)
    while not done:
        action = greedy(e.game, 1)[0] - 1
        next_obs, reward, done, _ = e.step(action)
        obs = next_obs

if __name__ == "__main__":
    # filename = generate_greedy(data_num=100000)
    tdata = np.load("./1528540963.npz")
    xs = tdata["xs"]
    ys = tdata["ys"]
    xs = xs[: 1304320]
    ys = ys[: 1304320]
    xs = np.reshape(xs, (5095, 256, 4, 6, 6))
    ys = np.reshape(ys, (5095, 256, 7))

    # model = ConvNet().cuda()
    model = torch.load("pre_cnn.pkl").cuda()
    model.train()

    validate(model)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3,
                        momentum=0.9, weight_decay=5e-4)

    for _ in range(100):
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(zip(xs, ys)), total=5095):
            inputs, targets = torch.FloatTensor(inputs), torch.FloatTensor(targets)
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        print(train_loss / 5095)
        torch.save(model, "pre_cnn.pkl")
        validate(model)
