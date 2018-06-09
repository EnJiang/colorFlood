from alpha_go_utils.data import *
from alpha_go_utils.resnet import ResNet18
import numpy as np


if __name__ == "__main__":
    filename = generate_greedy()
    data_set = MyDataset("./data/greedy_1/%s" % filename)
    model = ResNet18()