import torch 
import torch.nn as nn
import numpy as np

class ConvNet(nn.Module):
    def __init__(self, output_shape=7):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, output_shape)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        # print(out.size())
        # exit()
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)

        last_1 = self.softmax(out[:, : -1])
        last_2 = out[:, -1: ]
        out = torch.cat((last_1, last_2), 1)

        return out

class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func_1 = nn.CrossEntropyLoss()
        self.loss_func_2 = nn.MSELoss()

    def forward(self, predict, real):
        target_1 = torch.max(real[:, : -1], 1)[1]
        # print(target_1)
        target_2 = real[:, -1:]
        loss_1 = self.loss_func_1(predict[:, : -1], target_1)
        loss_2 = self.loss_func_2(predict[:, -1: ], target_2)

        return loss_1 + loss_2

class MaskPiLoss(MyLoss):
    def forward(self, predict, real):
        target_2 = real[:, -1:]
        loss_2 = self.loss_func_2(predict[:, -1:], target_2)

        return loss_2

if __name__ == "__main__":
    model = ConvNet()
    data = np.random.rand(1, 4, 11, 11)
    data = torch.Tensor(data)
    output = model(data)
    print(output)
