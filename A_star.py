#coding:utf-8

import random
import copy
from Game import *


class AStartSolver:
    def __init__(self):
        self.lowest_f_game = Game()
        self.counter = 0
        self.result = 255
        self.open = []
        self.close = set()
        self.explore()

    def run(self):
        done = False
        while not done:
            self.lowest_f_game = self.pop_lowest_f_game()
            self.close.add(self.lowest_f_game.hash_string())

            if self.lowest_f_game.isOver():
                done = True
                # self.report()
                self.result = self.lowest_f_game.step
            else:
                self.counter += 1
                self.report()
                self.explore()

    def pop_lowest_f_game(self):
        self.open.sort(reverse=True)
        lowest_f_game = self.open.pop()

        return lowest_f_game

    def report(self):
        if self.counter % 1000 == 0:
        # if True:
            print("iteration %d, f: %d, path: %s, step:%d"
                % (self.counter, self.lowest_f_game.f, self.lowest_f_game.allStep, self.lowest_f_game.step))
        # print(self.open)

    def explore(self):
        last_color = int(
            self.lowest_f_game.allStep[-1]) if len(self.lowest_f_game.allStep) else None
        color_to_pick = [i + 1 for i in range(6)]
        if last_color:
            color_to_pick.remove(last_color)

        for color in color_to_pick:
            copy_game = copy.deepcopy(self.lowest_f_game)
            copy_game.change(color)

            dupicate = copy_game.hash_string() in self.close

            if not dupicate:
                self.open.append(copy_game)


if __name__ == "__main__":
    result = []
    while 1:
        a = AStartSolver()
        a.run()
        result.append(a.result)
        print(sum(result) * 1.0 / len(result), len(result))
