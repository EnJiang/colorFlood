from alpha_go_utils.data import *
import numpy as np

if __name__ == "__main__":
    filename = generate_greedy()
    tdata = np.load("./data/greedy_1/%s" % filename)
    print(tdata["xs"])
