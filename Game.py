#coding:utf-8

import random
import copy
import numpy


class Node():
    def __init__(self, position, father):
        self.x, self.y = position
        self.father = father
        self.status = 1

    def explore(self, board, link):
        #确定下一个探索点,进入下一个状态
        x, y = self.x, self.y
        check = None
        if(self.status == 1 and self.x != 11):
            check = [x + 1, y]
        if(self.status == 2 and self.y != 11):
            check = [x, y + 1]
        if(self.status == 3 and self.x != 0):
            check = [x - 1, y]
        if(self.status == 4 and self.y != 0):
            check = [x, y - 1]
        self.status = self.status + 1

        #如果下一个探索点不存在或者下一个探索点已经在链表中,返回false,指示继续循环
        #否则,如果下一个探索点为同色,返回check,指示退出循环;否则返回false,指示继续循环
        if(check == None):
            #print self.status-1,self.x,',',self.y," edge"
            return False
        exist = False
        for n in link:
            if(n.x == check[0] and n.y == check[1]):
                exist = True
        if(exist):
            #print self.status-1,self.x,',',self.y," exist"
            return False
        if(board[0][0] == board[check[0]][check[1]]):
            #print self.status-1,self.x,',',self.y," child ",check
            return check
        else:
            #print self.status-1,self.x,',',self.y," not same"
            return False

    def next(self, board, link):
        check = False
        while (self.status != 5):
            check = self.explore(board, link)
            if(check):
                break
        return check


class Spider():
    def __init__(self, board):
        #定义链表,初始化蜘蛛在左上角第一个点
        self.link = [Node([0, 0], None)]
        self.spider = self.link[0]

    def clean(self):
        #所有节点为未检查
        for n in self.link:
            n.status = 1

    def nodeExist(self, child):
        #检查节点是不是已经存在于链表中
        exist = False
        for n in self.link:
            if (n.x == child.x and n.y == child.y):
                exist = True
        return exist

    def targetBoard(self, board):
        #定义链表,初始化蜘蛛在左上角第一个点
        self.link = [Node([0, 0], None)]
        self.spider = self.link[0]
        targetBoard = [[0 for x in range(12)] for y in range(12)]
        self.clean()
        next = self.spider.next(board, self.link)
        #只要不是spider在原点且检查完了所有方向(这意味着整个棋盘已经检查完毕),就继续检查
        while(not (self.spider.father == None and next == False)):
            if (next):    # 若是存在下一个节点,则新建这个节点插入到链表里面去(如果其不存在于链表中的话),蜘蛛来到新节点
                father = self.spider
                child = Node(next, father)
                if(not self.nodeExist(child)):
                    self.link.append(child)
                self.spider = child
            else:    # 否则就让蜘蛛回到父节点去
                self.spider = self.spider.father
            next = self.spider.next(board, self.link)
        #最后,将每一个节点所对应的(x,y坐标设为target point)
        for n in self.link:
            targetBoard[n.x][n.y] = 1
        return targetBoard


class Game():
    def __init__(self):
        #print node.__name__
        #init mainBoard
        self.mainBorad = [[0 for x in range(12)] for y in range(12)]
        posList = []
        for x in range(12):
            for y in range(12):
                posList.append([x, y])
        color = 1
        left = 143
        for x in range(6):
            for y in range(24):
                k = random.randint(0, left)
                i, j = posList[k]
                self.mainBorad[i][j] = color
                posList[k] = posList[left]
                left = left - 1
            color = color + 1
        # print self

        #init start
        self.start = ''
        for x in range(12):
            for y in range(12):
                self.start += str(self.mainBorad[x][y])

        #init all step
        self.allStep = ''

        #init spider
        self.spider = Spider(self.mainBorad)

        #init targetBoard
        self.targetBoard = self.spider.targetBoard(self.mainBorad)
        # print(self.targetBoard)
        #init step
        self.step = 0

        #init baseCorlor
        self.baseColor = self.mainBorad[0][0]

        self.f = 144 + 255

    def targetArea(self):
        area = 0
        for x in range(12):
            for y in range(12):
                if(self.targetBoard[x][y]):
                    area = area + 1
        return area

    def change(self, color):
        self.baseColor = color
        self.step = self.step + 1
        for x in range(12):
            for y in range(12):
                if(self.targetBoard[x][y] == 1):
                    self.mainBorad[x][y] = color
        #print self
        self.allStep += str(color)
        self.targetBoard = self.spider.targetBoard(self.mainBorad)
        #print self.targetBoard

        self.cal_f()

    def cal_f(self):
        remain_color = []
        for i in range(12):
            for j in range(12):
                if self.mainBorad[i][j] not in remain_color:
                    remain_color.append(self.mainBorad[i][j])

        smallest_manhattan_distance = 24
        for x in range(12):
            for y in range(12):
                if(self.targetBoard[x][y] == 1):
                    manhattan_distance = 11 - x + 11 - y
                    if manhattan_distance < smallest_manhattan_distance:
                        smallest_manhattan_distance = manhattan_distance

        self.f = self.step + max(smallest_manhattan_distance, len(remain_color) - 1) + \
                 (144 - self.targetArea()) * 1.0 / 144
        # self.f = self.step + smallest_manhattan_distance + \
            # len(remain_color) - 1 + (144 - self.targetArea()) * 1.0 / 144


    def __eq__(self, game):
        for i in range(12):
            for j in range(12):
                if self.mainBorad[i][j] != game.mainBorad[i][j]:
                    return False
        return True

    def __lt__(self, game):
        return self.f < game.f

    def isOver(self):
        if (self.targetArea() == 144):
            return True
        else:
            return False

    def mainBoardArray(self):
        return numpy.array(
            [
                numpy.array(self.mainBorad).reshape(144)
            ]
        )

    def mainBoardMatrix(self):
        return numpy.array(
            [
                numpy.array(self.mainBorad).reshape(12, 12, 1)
            ]
        )

    def mainBoardString(self):
        string = ""
        for i in self.mainBorad:
            for j in i:
                string += str(j)
        return string

    def colorDistriString(self):
        colorDistri = []
        # print self.mainBorad
        for i in range(6):
            color = i + 1
            temp = ""
            for j in self.mainBorad:
                for k in j:
                    if(k == color):
                        temp += '1'
                    else:
                        temp += '0'
            # print temp
            colorDistri.append(temp)
        string = ""
        # print sorted(colorDistri)
        for s in sorted(colorDistri):
            string += s
        return string

    def initByString(self, string):
        for i in range(12):
            for j in range(12):
                self.mainBorad[i][j] = int(string[i * 12 + j])
        self.targetBoard = self.spider.targetBoard(self.mainBorad)

    def hash_string(self):
        output = ''
        for x in range(12):
            for y in range(12):
                output += str(self.mainBorad[x][y])
        return output

    def __str__(self):
        output = ''
        for x in range(12):
            for y in range(12):
                if (y != 11):
                    output += str(self.mainBorad[x][y]) + "    "
                else:
                    output += str(self.mainBorad[x][y])
            output += "\n"
        return output



if __name__ == "__main__":
    game = Game()

