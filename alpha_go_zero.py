from alpha_go_utils.data import *
from alpha_go_utils.network import ConvNet
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from env import Env
from torch.utils.data import DataLoader

def validate(model):
    print("\n\n\n\n\n\n")
    print("--------validating--------")
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
        print(action_greedy, action_pi)

        next_obs, reward, done, _ = e.step(action_greedy)
        obs = next_obs
        obs = np.reshape(obs, (1, 4, 6, 6))
        obs = torch.FloatTensor(obs).cuda()
    print("\n\n\n\n\n\n")


if __name__ == "__main__":
    # filename = generate_greedy(data_num=100000)
    tdata = np.load("./1528531033.npz")
    xs = tdata["xs"]
    ys = tdata["ys"]

    data_set = MyDataset(xs, ys)
    dataloader = DataLoader(data_set, batch_size=32, shuffle=True)

    model = ConvNet().cuda()
    # model = torch.load("pre_cnn.pkl").cuda()
    model.train()

    # validate(model)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3,
                        momentum=0.9, weight_decay=5e-4)

    for epoch in range(100):
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, sample in tqdm(enumerate(dataloader), total=len(data_set) // 32):
            # print(type(sample))
            inputs, targets = sample[0], sample[1]
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        print(train_loss / 5095)
        torch.save(model, "pre_cnn.pkl")

        if epoch % 3 == 2:
            validate(model)
