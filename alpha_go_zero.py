from alpha_go_utils.data import *
from alpha_go_utils.network import ConvNet, MyLoss
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from env import Env
from torch.utils.data import DataLoader


def validate(model, vdataloader, tbatch_num, criterion):
    model.eval()
    train_loss = 0
    for batch_idx, sample in tqdm(enumerate(vdataloader), total=tbatch_num):
        # print(type(sample))
        inputs, targets = sample[0], sample[1]
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.item()
    print(train_loss / tdata_size)
    model.train()

def test(model):
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
    batch_size = 256

    # filename = generate_greedy(data_num=100000)
    # tdata = np.load("./1528760107.npz")
    txs = np.load(open("txs.npy", "rb"))
    tys = np.load(open("tys.npy", "rb"))

    # vdata = np.load("./1528531033.npz")
    vxs = np.load(open("vxs.npy", "rb"))
    vys = np.load(open("vys.npy", "rb"))

    tdata_size = len(txs)
    tbatch_num = len(txs) // batch_size

    tset = MyDataset(txs, tys)
    vset = MyDataset(vxs, vys)
    tdataloader = DataLoader(tset, batch_size=batch_size, shuffle=True)
    vdataloader = DataLoader(vset, batch_size=batch_size, shuffle=False)

    model = ConvNet().cuda()
    # model = torch.load("pre_cnn.pkl").cuda()
    model.train()

    # validate(model)

    criterion = MyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3,
                        momentum=0.9, weight_decay=5e-4)

    for epoch in range(100):
        train_loss = 0
        for batch_idx, sample in tqdm(enumerate(tdataloader), total=tbatch_num):
            # print(type(sample))
            inputs, targets = sample[0], sample[1]
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        print(train_loss / tdata_size)

        if epoch % 3 == 2:
            validate(model, vdataloader, len(vxs) // batch_size, criterion)
            torch.save(model, "pre_cnn.pkl")
