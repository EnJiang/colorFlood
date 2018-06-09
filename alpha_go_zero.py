from alpha_go_utils.data import *
from alpha_go_utils.network import ConvNet
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

if __name__ == "__main__":
    filename = generate_greedy(data_num=100000)
    # tdata = np.load("./1528531033.npz")
    # xs = tdata["xs"]
    # ys = tdata["ys"]
    # xs = np.reshape(xs, (1305, 10, 4, 6, 6))
    # ys = np.reshape(ys, (1305, 10, 7))

    # model = ConvNet().cuda()
    # model.train()

    # criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-3,
    #                     momentum=0.9, weight_decay=5e-4)

    # for _ in range(100):
    #     train_loss = 0
    #     correct = 0
    #     total = 0
    #     for batch_idx, (inputs, targets) in tqdm(enumerate(zip(xs, ys))):
    #         inputs, targets = torch.FloatTensor(inputs), torch.FloatTensor(targets)
    #         inputs, targets = inputs.cuda(), targets.cuda()
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()

    #         train_loss += loss.item()
    #     print(train_loss)
