# coding:utf-8

import random
import copy
from collections import namedtuple
from bisect import insort_left, bisect_left
from game import *
from time import time
import pymysql
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    db = pymysql.connect(host='',
                         port=3306,
                         user='root',
                         password='xxxxxxx',
                         db='colorflood',
                         charset='utf8')
except:
    db = None

Record = namedtuple("Record", "f games")
def __lt__(self, other):
    if isinstance(other, Record):
        return self.f < other.f
    else:
        return self.f < other
Record.__lt__ = __lt__


class AStartSolver:
    def __init__(self):
        self.lowest_f_game = Game(need_cal_f=True)
        self.init_state = self.lowest_f_game.hash_string()
        self.counter = 0
        self.result = 255
        self.open = []
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
                # self.report()
                self.result = self.lowest_f_game.step
            else:
                self.counter += 1
                # self.report()
                self.explore()
        print("done in %d path" % self.lowest_f_game.step)
        # print(time() - start, self.counter)

    def save(self):
        if db is None:
            return self.lowest_f_game.step

        sql = "INSERT INTO a_star_171118(init_state, step, path) VALUES(\'%s\', \'%s\', \'%s\')" % (
            self.init_state, self.lowest_f_game.step, self.lowest_f_game.allStep
        )
        curs = db.cursor()
        curs.execute(sql)
        db.commit()
        return self.lowest_f_game.step

    def pop_lowest_f_game(self):
        # print(self.open)
        lowest_f = self.open[0].f
        lowest_f_game = self.open[0].games.pop()

        if len(self.open[0].games) == 0:
            del self.open[0]

        return lowest_f_game

    def report(self):
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

            # print(self.open)

            if not dupicate:
                f = copy_game.f
                i = bisect_left(self.open, f)
                if not i and not len(self.open): # special case
                    record = Record(f, [copy_game])
                    insort_left(self.open, record)

                if self.open[i].f == f:
                    self.open[i].games.append(copy_game)
                else:
                    record = Record(f, [copy_game])
                    insort_left(self.open, record)


def one():
    a = AStartSolver()
    a.run()
    step = a.save()
    f = open("result_a_star.txt", "a+")
    f.write(str(step) + '\n')
    f.close()


if __name__ == "__main__":
    executor = ProcessPoolExecutor(max_workers=8)
    for _ in range(8):
        executor.submit(one)
