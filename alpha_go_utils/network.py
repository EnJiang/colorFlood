import torch 
import torch.nn as nn
import numpy as np

class ConvNet(nn.Module):
    def __init__(self, output_shape=7):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_shape)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
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
        self.loss_func_2 = nn.BCELoss()

    def forward(self, predict, real):
        target_1 = torch.LongTensor(real[:, : -1])
        target_2 = torch.LongTensor(real[:, -1:])
        loss_1 = self.loss_func_1(predict[:, : -1], target_1)
        loss_2 = self.loss_func_2(predict[:, -1: ], target_2)

        return loss_1 + loss_2

if __name__ == "__main__":
    model = ConvNet()
    data = np.random.rand(1, 4, 11, 11)
    data = torch.Tensor(data)
    output = model(data)
    print(output)
