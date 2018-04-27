# coding:utf-8

import random
import copy
from Game import *


def getRand(last):
    color = random.randint(1, 6)
    while color == last:
        color = random.randint(1, 6)
    return color


while 1:
    game = Game()
    last = 0
    while not game.isOver():
        color = getRand(last)
        last = color
        game.change(color)
    print game.step, game.allStep
