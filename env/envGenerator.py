import numpy as np
import matplotlib.pyplot as plt
import math

class Env():
    """Definition of random environment generation"""
    def __init__(self, center, radius, cellsize, width, height):
        self.x_n = int(width/cellsize)
        self.y_n = int(height/cellsize)
        # gen map matrix
        self.cellsize = cellsize
        self.grid = np.zeros((self.y_n, self.x_n))

        self._gen_circle(radius, center)
        self._place_goal()

    def _gen_circle(self, radius, center):
        # generate open circle 
        for yi in range(self.y_n):
            for xi in range(self.x_n):
                # check if cell center is within bounds
                xpos = xi*self.cellsize+self.cellsize/2
                ypos = yi*self.cellsize+self.cellsize/2
                dist = ((xpos-center[0])**2+(ypos-center[1])**2)**0.5
                if dist >= radius:
                    self.grid[yi, xi] = 1
    
    def _place_goal(self):
        done = False
        while not done:
            xi = int(np.random.rand()*self.x_n)
            yi = int(np.random.rand()*self.y_n)
            if self.grid[yi, xi] == 0:
                self.grid[yi, xi] = 2
                done = True

    def draw_map(self):
        plt.figure(0)
        plt.title("Map Layout")
        plt.imshow(self.grid)
        plt.show()

    def validate(self, agent):
        xi = math.ceil(agent.x/self.cellsize)
        yi = math.ceil(agent.y/self.cellsize)
        return self.grid[yi, xi]