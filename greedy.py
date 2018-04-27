# coding:utf-8

from game import *
# import MySQLdb
import copy
import time


def toTry(depth):
    tryList = [[] for x in range(pow(6, depth))]
    for index in range(len(tryList)):
        tIndex = copy.copy(index)
        for power in range(depth):
            step = int(tIndex / pow(6, depth - power - 1))
            tryList[index].append(step + 1)
            tIndex = tIndex - step * pow(6, depth - power - 1)
        final = (index + 1) % 6
        if(final == 0):
            final = 6
        tryList[index][-1] = final

    for tryIndex in range(len(tryList)):
        for stepIndex in range(len(tryList[tryIndex])):
            if (stepIndex == 0):
                continue
            if (tryList[tryIndex][stepIndex] == tryList[tryIndex][stepIndex - 1]):
                tryList[tryIndex] = [0 for y in range(depth)]
    return tryList


def greedy(game, depth):
    improve = 0
    area = game.targetArea()
    tryList = toTry(depth)
    for aTry in tryList:
        if (aTry[0] == 0):
            continue
        tempGame = copy.deepcopy(game)
        improveEachTemp = []
        for step in aTry:
            tempGame.change(step)
            improveEachTemp.append(tempGame.targetArea() - area)
        if (tempGame.targetArea() - area > improve):
            improve = tempGame.targetArea() - area
            improveEach = improveEachTemp
            color = aTry
    # print improveEach
    for iEIndex in range(depth):
        if(improveEach[depth - iEIndex - 1] == 0):
            del color[depth - iEIndex - 1]
    # print color
    return color

if __name__ == "__main__":
    result = []
    depth = 1
    while 1:
        start_time = time.time()
        game = Game()
        init = game.hash_string()
        while not game.isOver():
            color = greedy(game, depth)
            for c in color:
                if not game.isOver():
                    game.change(c)
                    # game.allStep+=str(c)
                else:
                    break
        step = len(game.allStep)
        allStep = game.allStep
        # print "use",step,"step:",allStep

        result.append(step)
        print(sum(result) * 1.0 / len(result),
            len(result), time.time() - start_time)
