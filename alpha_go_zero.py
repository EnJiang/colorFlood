from alpha_go_utils.data import *
from alpha_go_utils.resnet import ResNet18
import numpy as np
import torch.optim as optim
import torch.nn as nn

if __name__ == "__main__":
    # filename = generate_greedy(data_num=1000)
    data_set = MyDataset("./1528531033.npz")

    model = ResNet18().cuda()
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3,
                        momentum=0.9, weight_decay=5e-4)

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_set):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
