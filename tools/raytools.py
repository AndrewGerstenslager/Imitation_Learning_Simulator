import numpy as np
import math

def wrapTheta(theta):
    """
    wrap theta within bounds [0-360)
    """
    rotations = math.floor(theta / 360)
    wrapped = theta - rotations*360
    return wrapped

def getGridIndex(cellsize, x, y):
    """
    gets grid index of cell at position (x, y)
    """
    xi = int(math.floor(x/cellsize))
    yi = int(math.floor(y/cellsize))
    return [xi, yi]

def getCellPos(cellsize, x, y):
    """
    gets position within cell at position (x, y)
    """
    x_pos = x - cellsize*math.floor(x/cellsize)
    y_pos = y - cellsize*math.floor(y/cellsize)
    return [x_pos, y_pos]

def getCollision(cellsize, grid, x, y):
    ylen = len(grid[:,0])-1
    xlen = len(grid[0,:])-1
    idx = getGridIndex(cellsize, x, y)
    if idx[0] < 0 or idx[0] > xlen:
        return False
    if idx[1] < 0 or idx[1] > ylen:
        return False
    else:
        return grid[idx[1], idx[0]] == 1 # this is flipped...

def norm(x, y):
    return (x**2 + y**2)**0.5