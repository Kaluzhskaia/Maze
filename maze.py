# -*- coding: utf-8 -*-
import numpy
import copy
import time
from numpy.random import random_integers as rand
from matplotlib import colors
import matplotlib.colors as col
import matplotlib.pyplot as plt
import matplotlib.animation as animation



fig = plt.figure(figsize=(8, 8))


class Maze:
    mazeCopy = []
    wavePath = []
    imageList = []

    def update(self, i):
        im.set_array(self.imageList[i])
        return im,

    def __init__(self, width=81, height=51, complexity=.05, density=.9):

        self.shape = ((width // 2) * 2 + 1, (height // 2) * 2 + 1)  # (x, y)

        self.complexity = int(complexity * (5 * (self.shape[0] + self.shape[1])))
        self.density = int(density * ((self.shape[0] // 2) * (self.shape[1] // 2)))
        # build maze
        self.Z = numpy.zeros(self.shape, dtype=float)


        self.Z[0, :] = self.Z[-1, :] = 1
        self.Z[:, 0] = self.Z[:, -1] = 1

        for i in range(self.density):
            x, y = rand(0, self.shape[0] // 2) * 2, rand(0, self.shape[1] // 2) * 2
            self.Z[x, y] = 1
            for j in range(self.complexity):
                neighbours = []
                if x > 1:             neighbours.append((x - 2, y))
                if x < self.shape[0] - 2:  neighbours.append((x + 2, y))
                if y > 1:             neighbours.append((x, y - 2))
                if y < self.shape[1] - 2:  neighbours.append((x, y + 2))
                if len(neighbours):
                    x_, y_ = neighbours[rand(0, len(neighbours) - 1)]
                    if self.Z[x_, y_] == 0:
                        self.Z[x_, y_] = 1  # black
                        self.Z[x_ + (x - x_) // 2, y_ + (y - y_) // 2] = 1
                        x, y = x_, y_  # перейти к соседу

        self.mazeCopy = self.Z.copy()

    def genMaze(self):
        self.Z = numpy.zeros(self.shape, dtype=float)

        self.imageList.append(self.Z.tolist())

        self.Z[0, :] = self.Z[-1, :] = -1
        self.Z[:, 0] = self.Z[:, -1] = -1
        self.imageList.append(self.Z.tolist())


        for i in range(self.density):
            x, y = rand(0, self.shape[0] // 2) * 2, rand(0, self.shape[1] // 2) * 2
            self.Z[x, y] = 1
            self.imageList.append(self.Z.tolist())
            self.Z[x, y] = -1
            for j in range(self.complexity):
                neighbours = []
                if x > 1:             neighbours.append((x - 2, y))
                if x < self.shape[0] - 2:  neighbours.append((x + 2, y))
                if y > 1:             neighbours.append((x, y - 2))
                if y < self.shape[1] - 2:  neighbours.append((x, y + 2))
                if len(neighbours):
                    x_, y_ = neighbours[rand(0, len(neighbours) - 1)]
                    #tempVal = self.Z[x_, y_]
                    #self.Z[x_, y_] = 1

                    #self.imageList.append(self.Z.tolist())
                    #self.Z[x_, y_] = tempVal
                    if self.Z[x_, y_] == 0:
                        self.Z[x, y] = 1
                        self.Z[x_, y_] = -1  # black
                        self.imageList.append(self.Z.tolist())
                        self.Z[x_ + (x - x_) // 2, y_ + (y - y_) // 2] = -1
                        self.imageList.append(self.Z.tolist())
                        self.Z[x, y] = -1
                        x, y = x_, y_  # go to neighbour

        self.Z[1, 0] = self.Z[-2, -1] = 0
        self.imageList.append(self.Z.tolist())

        self.mazeCopy = self.Z.copy()
        global im
        im = plt.imshow(self.imageList[0], cmap="RdGy_r", interpolation='nearest', vmin=-1, vmax=1)
        ani = animation.FuncAnimation(fig, self.update, frames=range(len(self.imageList)), interval=100, blit=True)
        plt.show()



    def drawWavePath(self):
        for item in self.wavePath:
            self.Z[item] = 2
        self.Z[1, 0] = 2
        self.Z[-2, -1] = 2


    def refresh(self):
        self.Z = self.mazeCopy.copy()

    def startWave(self, sX, sY, fX, fY):
        bufMaze = self.mazeCopy.copy()
        for i in range(0, self.shape[0]):
            for j in range(0, self.shape[1]):
                if bufMaze[i, j] == 1: bufMaze[i, j] = 111
        # start
        bufMaze[sX, sY] = 0
        # finish
        bufMaze[fX, fY] = -2
        self.waveForward(bufMaze, sX, sY, 1)
        # start cell is 0 again
        bufMaze[sX, sY] = 0
        # go back and track the route
        self.wavePath.append((fX, fY))
        self.waveBack(bufMaze, fX, fY, self.wavePath)
        #print bufMaze
        print "Волновой алгоритм"
        print "Кратчайший найденный путь:"
        print len(self.wavePath)

    def waveForward(self, bufMaze, sX, sY, d):
        # if finish cell found mark it and break
        if bufMaze[sX - 1, sY] == -2:
            bufMaze[sX - 1, sY] = d
            return
        if bufMaze[sX, sY - 1] == -2:
            bufMaze[sX, sY - 1] = d
            return
        if bufMaze[sX + 1, sY] == -2:
            bufMaze[sX + 1, sY] = d
            return
        if bufMaze[sX, sY + 1] == -2:
            bufMaze[sX, sY + 1] = d
            return

        #
        if bufMaze[sX - 1, sY] == 0:
            bufMaze[sX - 1, sY] = d
        if bufMaze[sX, sY - 1] == 0:
            bufMaze[sX, sY - 1] = d
        if bufMaze[sX + 1, sY] == 0:
            bufMaze[sX + 1, sY] = d
        if bufMaze[sX, sY + 1] == 0:
            bufMaze[sX, sY + 1] = d

        # and go into them
        if bufMaze[sX - 1, sY] == d:
            self.waveForward(bufMaze, sX - 1, sY, d + 1)
        if bufMaze[sX, sY - 1] == d:
            self.waveForward(bufMaze, sX, sY - 1, d + 1)
        if bufMaze[sX + 1, sY] == d:
            self.waveForward(bufMaze, sX + 1, sY, d + 1)
        if bufMaze[sX, sY + 1] == d:
            self.waveForward(bufMaze, sX, sY + 1, d + 1)

    def waveBack(self, bufMaze, fX, fY, a):
        # add elements which are 1 smaller when the finish cell
        # and go into them
        if bufMaze[fX, fY] == bufMaze[fX - 1, fY] + 1:
            a.append((fX - 1, fY))
            self.waveBack(bufMaze, fX - 1, fY, a)
        if bufMaze[fX, fY] == bufMaze[fX, fY - 1] + 1:
            a.append((fX, fY - 1))
            self.waveBack(bufMaze, fX, fY - 1, a)
        if bufMaze[fX, fY] == bufMaze[fX + 1, fY] + 1:
            a.append((fX + 1, fY))
            self.waveBack(bufMaze, fX + 1, fY, a)
        if bufMaze[fX, fY] == bufMaze[fX, fY + 1] + 1:
            a.append((fX, fY + 1))
            self.waveBack(bufMaze, fX, fY + 1, a)



    def findNeighbours(self, someMaze, x, y):
        neighbours = []
        if someMaze[x + 1, y] != -1:
            neighbours.append((x + 1, y))
        if someMaze[x - 1, y] != -1:
            neighbours.append((x - 1, y))
        if someMaze[x, y + 1] != -1:
            neighbours.append((x, y + 1))
        if someMaze[x, y - 1] != -1:
            neighbours.append((x, y - 1))

        return neighbours

    def maxEl(self, maze):
        max = 0
        for i in range(0, maze.shape[0]):
            for j in range(0, maze.shape[1]):
                if maze[i, j] >= max: max = maze[i, j]
        return max

    def RFD(self, sX, sY, fX, fY):
        kol = 0
        while kol < 3:
            kol = 0
            riverMaze = self.mazeCopy.copy()
            area = riverMaze.shape[0] * riverMaze.shape[1]
            for i in range(0, riverMaze.shape[0]):
                for j in range(0, riverMaze.shape[1]):
                    if riverMaze[i, j] == 1: riverMaze[i, j] = -1
                    # set beginning values
                    if riverMaze[i, j] == 0: riverMaze[i, j] = .5

            # lets make iterations
            for i in range(0, riverMaze.shape[0] * riverMaze.shape[1] * 60):
                print i, kol
                track = []
                curX, curY = sX, sY
                prevX, prevY = curX, curY
                while (1):
                    # if ant makes a loop and comes to the start cell again
                    if len(track) > 1 and (curX, curY) == (sX, sY): break
                    # tracking the route
                    track.append((curX, curY))
                    neighbours = self.findNeighbours(riverMaze, curX, curY)

                    if len(neighbours) == 1 and (curX, curY) != (prevX, prevY): break

                    # if finish reached
                    if (curX, curY) == (fX, fY):
                        kol += 1
                        for m in range(0, len(track) - 1):
                            riverMaze[track[m]] -= (1.0 / len(track)) * 4
                            if riverMaze[track[m]] <= 0: riverMaze[track[m]] = 0.00001

                            nei = self.findNeighbours(riverMaze, track[m][0], track[m][1])
                            if m != 0:
                                previous = track[m-1]
                                nei.remove(previous)
                            nextM = track[m + 1]
                            nei.remove(nextM)
                            for n in nei:
                                riverMaze[n] += (1.0/len(track)) * 10
                                if riverMaze[n] >= 1: riverMaze[n] = .99
                        break




                    # if not standing at start cell, remove prev cell from list of possible cells to run to
                    if (curX, curY) != (prevX, prevY): neighbours.remove((prevX, prevY))
                    # make prev currect and move to a random neighbour
                    prevX, prevY = curX, curY
                    p = []
                    sum = 0
                    for each in neighbours:
                        sum += (1 - riverMaze[each])
                    for each in neighbours:
                        p.append((1 - riverMaze[each]) / sum)

                    numbers = numpy.arange(len(neighbours))
                    cell = numpy.random.choice(numbers, 1, p)
                    curX, curY = neighbours[cell][0], neighbours[cell][1]


        riverMaze[fX, fY] = 0
        curX, curY = sX, sY
        prevX, prevY = curX, curY
        lenght = 0


        backpath = []
        backpath.append((1,0))
        backpath.append((-2,-1))
        backpath.append((sX, sY))

        while (1):
            if (curX, curY) == (fX, fY): break
            lenght += 1
            neighbours = self.findNeighbours(riverMaze, curX, curY)
            if (curX, curY) != (prevX, prevY): neighbours.remove((prevX, prevY))
            minI = neighbours[0]
            for each in neighbours:
                if riverMaze[each] <= riverMaze[minI]: minI = each

            prevX, prevY = curX, curY
            backpath.append((minI[0], minI[1]))
            curX, curY = minI[0], minI[1]


        fields = numpy.empty(riverMaze.shape)* numpy.nan
        for each in backpath:
            fields[each] = 1

        print kol

        riverMaze[1, 0] = 0
        riverMaze[-2, -1] = 0
        numpy.set_printoptions(precision=3, suppress=True, threshold=numpy.nan, linewidth=300)

        return riverMaze, fields, lenght



    def antColony(self, sX, sY, fX, fY):
        kol = 0
        while kol < 3:
            kol = 0
            antMaze = self.mazeCopy.copy()
            square = antMaze.shape[0] * antMaze.shape[1]
            for i in range(0, antMaze.shape[0]):
                for j in range(0, antMaze.shape[1]):
                    if antMaze[i, j] == 1: antMaze[i, j] = -1
                    # set beginning pheromone values
                    if antMaze[i, j] == 0: antMaze[i, j] = 0.2

            # lets make iterations
            for i in range(0, square * 60):
                print i, kol
                track = []
                curX, curY = sX, sY
                prevX, prevY = curX, curY
                while (1):
                    # if ant makes a loop and comes to the start cell again
                    if len(track) > 1 and (curX, curY) == (sX, sY): break
                    # tracking the route
                    track.append((curX, curY))
                    neighbours = self.findNeighbours(antMaze, curX, curY)

                    # if finish reached, update all path to it as more attractive
                    if (curX, curY) == (fX, fY):
                        kol += 1
                        for item in track:
                            antMaze[item] += numpy.exp(-len(track)/square) / 3.0
                            # if above 1, make it super attractive
                            if antMaze[item] >= 1: antMaze[item] = 0.99
                        #self.imageList.append(antMaze.tolist())
                        break

                    # if we are in a dead end
                    # so we have just one neighbour-prev cell(and we check its not the beggining)
                    # update all paths to it as unattractive
                    if len(neighbours) == 1 and (curX, curY) != (prevX, prevY):
                        track.reverse()
                        for item in track:
                            # find the first crossing from the dead end
                            # and update the pheromones on it
                            if len(self.findNeighbours(antMaze, item[0], item[1])) >= 3:
                                deadend = track[:track.index(item)]
                                for cell in deadend:
                                    antMaze[cell] -= 1.0 / len(track) * (square / 20.0)
                                    # if value accidentelly goes below zero, update its pheromone as super unattractive
                                    if antMaze[cell] <= 0: antMaze[cell] = 0.00001
                                break
                        #self.imageList.append(antMaze.tolist())
                        break

                    # if not standing at start cell, remove prev cell from list of possible cells to run to
                    if (curX, curY) != (prevX, prevY): neighbours.remove((prevX, prevY))
                    # make prev currect and move to a random neighbour
                    prevX, prevY = curX, curY
                    p = []
                    sum = 0
                    for each in neighbours:
                        sum += antMaze[each]
                    for each in neighbours:
                        p.append(antMaze[each] / sum)

                    numbers = numpy.arange(len(neighbours))
                    cell = numpy.random.choice(numbers, 1, p)
                    curX, curY = neighbours[cell][0], neighbours[cell][1]

                # evaporate
                for k in range(0, antMaze.shape[0]):
                    for l in range(0, antMaze.shape[1]):
                        if antMaze[k, l] != -1:
                            #antMaze[k, l] -= 1.0 / (square * 39.0)
                            antMaze[k, l] -= 1.0 / square
                            if antMaze[k, l] <= 0: antMaze[k, l] = 0.00001

            print kol



        curX, curY = fX, fY
        print curX, curY
        prevX, prevY = curX, curY
        lenght = 0

        backpath = []
        backpath.append((1, 0))
        backpath.append((-2, -1))
        backpath.append((sX, sY))
        backpath.append((fX, fY))

        while (1):
            if (curX, curY) == (sX, sY): break
            lenght += 1
            neighbours = self.findNeighbours(antMaze, curX, curY)
            print neighbours
            if (curX, curY) != (prevX, prevY): neighbours.remove((prevX, prevY))
            maxI = neighbours[0]
            for each in neighbours:
                if antMaze[each] >= antMaze[maxI]: maxI = each

            prevX, prevY = curX, curY
            backpath.append((maxI[0], maxI[1]))
            curX, curY = maxI[0], maxI[1]

        fields = numpy.empty(antMaze.shape) * numpy.nan
        for each in backpath:
            fields[each] = 1

        #print "Алгоритм муравьиной колонии"
        #print "Кратчайший найденный путь:"
        #print lenght
        antMaze[1, 0] = 1
        antMaze[-2, -1] = 1
        numpy.set_printoptions(precision=3, suppress=True, threshold=numpy.nan, linewidth=300)
        #print antMaze
        return antMaze, fields, lenght


def maximum(maze):
    maxi = 0
    for i in range(0, maze.shape[0]):
        for j in range(0, maze.shape[1]):
            if maze[i, j] >= maxi:
                maxi = maze[i, j]
    return maxi


newMaze = Maze(20, 20)

#newMaze.genMaze()

my_cmap = copy.copy(plt.cm.get_cmap('Greens_r'))
my_cmap.set_bad(alpha=0)



fig0 = plt.figure(figsize=(10, 5))
start0 = time.time()
antMaze, antFields, lenghtA = newMaze.antColony(1, 1, newMaze.shape[0] - 2, newMaze.shape[1] - 2)
end0 = time.time()
plt.imshow(antMaze, cmap="RdGy_r", interpolation="nearest", vmin=-1, vmax=1)
plt.imshow(antFields, cmap=my_cmap, interpolation="nearest", alpha=0.8)
fig00 = plt.figure(figsize=(10, 5))
plt.imshow(antMaze, cmap="RdGy_r", interpolation="nearest", vmin=-1, vmax=1)

fig1 = plt.figure(figsize=(10, 5))
start1 = time.time()
riverMaze, riverFields, lenghtR = newMaze.RFD(1, 1, newMaze.shape[0] - 2, newMaze.shape[1] - 2)
end1 = time.time()
plt.imshow(riverMaze, cmap="RdGy_r", interpolation="nearest", vmin=-1, vmax=1)
plt.imshow(riverFields, cmap=my_cmap, interpolation="nearest", alpha=0.8)
fig11 = plt.figure(figsize=(10, 5))
plt.imshow(riverMaze, cmap="RdGy_r", interpolation="nearest", vmin=-1, vmax=1)

print "Размер лабиринта:"
print newMaze.shape
print

print "Алгоритм муравьиной колонии"
print "Кратчайший найденный путь:"
print lenghtA
print "Время работы:"
print(end0 - start0)
print

print "Алгоритм формирования рек"
print "Кратчайший найденный путь:"
print lenghtR
print "Время работы:"
print(end1 - start1)
print

fig2 = plt.figure(figsize=(10, 5))
start2 = time.time()
newMaze.startWave(1, 1, newMaze.shape[0]-2, newMaze.shape[1]-2)
newMaze.drawWavePath()
end2 = time.time()
print "Время работы:"
print(end2 - start2)
print
plt.imshow(newMaze.Z, cmap=colors.ListedColormap(["white", "black", "red"]), interpolation='nearest')


#plt.xticks([]), plt.yticks([])
plt.show()
