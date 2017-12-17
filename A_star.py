#coding:utf-8

import random
import copy
from Game import *
from time import time
import pymysql

db = pymysql.connect(host='114.115.217.207',
                             port=3306,
                             user='root',
                             password='123000000z',
                             db='colorflood',
                             charset='utf8')


class AStartSolver:
    def __init__(self):
        self.lowest_f_game = Game()
        self.init_state = self.lowest_f_game.hash_string()
        self.counter = 0
        self.result = 255
        self.open = {}
        self.close = set()
        self.explore()

    def run(self):
        done = False
        while not done:
            # start = time()
            self.lowest_f_game = self.pop_lowest_f_game()
            self.close.add(self.lowest_f_game.hash_string())

            if self.lowest_f_game.isOver():
                done = True
                self.report()
                self.result = self.lowest_f_game.step
            else:
                self.counter += 1
                self.report()
                self.explore()
        print("done in %d path" % self.lowest_f_game.step)
            # print(time() - start, self.counter)

    def save(self):
        sql = "INSERT INTO a_star_171118(init_state, step, path) VALUES(\'%s\', \'%s\', \'%s\')" % (
            self.init_state, self.lowest_f_game.step, self.lowest_f_game.allStep
        )
        curs = db.cursor()
        curs.execute(sql)
        db.commit()

    def pop_lowest_f_game(self):
        # print(self.open)
        lowest_f = min(self.open.keys())
        lowest_f_game = self.open[lowest_f].pop()

        if len(self.open[lowest_f]) == 0:
            self.open.pop(lowest_f)

        return lowest_f_game

    def report(self):
        return
        if self.counter % 1000 == 0:
        # if True:
            print("iteration %d, f: %f, path: %s, step:%d"
                % (self.counter, self.lowest_f_game.f, self.lowest_f_game.allStep, self.lowest_f_game.step))
            print(self.lowest_f_game)
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
                f = copy_game.f
                if f in self.open:
                    self.open[f].append(copy_game)
                else:
                    self.open[f] = [copy_game]


if __name__ == "__main__":
    result = []
    while 1:
        a = AStartSolver()
        a.run()
        f = open("result_1.txt", "a+")
        f.write(str(a.lowest_f_game.step) + '\n')
        f.close()
        # a.save()

