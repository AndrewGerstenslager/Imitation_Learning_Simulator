import numpy as np
import matplotlib.pyplot as plt
import math

class Env():
    """Definition of random environment generation"""
    def __init__(self, center, outer_radius, inner_radius, cellsize, width, height):
        self.x_n = int(width/cellsize)
        self.y_n = int(height/cellsize)
        # gen map matrix
        self.cellsize = cellsize
        self.grid = np.zeros((self.y_n, self.x_n))
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.center = center
        self._gen_circle()
        self._place_goal()

    def _gen_circle(self):
        # generate donut shape
        for yi in range(self.y_n):
            for xi in range(self.x_n):
                # check if cell center is within bounds
                xpos = xi*self.cellsize+self.cellsize/2
                ypos = yi*self.cellsize+self.cellsize/2
                dist = ((xpos-self.center[0])**2+(ypos-self.center[1])**2)**0.5
                if dist >= self.outer_radius or (dist <= self.inner_radius and dist > 0):  # Check for the inner radius
                    self.grid[yi, xi] = 1
    
    def _place_goal(self):
        done = False
        while not done:
            xi = int(np.random.rand()*self.x_n)
            yi = int(np.random.rand()*self.y_n)
            xpos = xi*self.cellsize+self.cellsize/2
            ypos = yi*self.cellsize+self.cellsize/2
            dist = ((xpos-self.center[0])**2+(ypos-self.center[1])**2)**0.5
            if self.grid[yi, xi] == 0 and dist > self.inner_radius and dist < self.outer_radius:
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