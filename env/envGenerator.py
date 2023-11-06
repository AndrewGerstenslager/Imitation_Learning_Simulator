import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import PIL
import tools.raytools as raytools
import pygame
sys.path.append("..")
import agent.turtle

class Env():
    """Definition of environment generation"""
    def __init__(self, param, roomtype, cellsize, width, height):
        self.x_n = int(width/cellsize)
        self.y_n = int(height/cellsize)
        # gen map matrix
        self.cellsize = cellsize
        self.grid = np.zeros((self.y_n, self.x_n))
        self.param = param
        if roomtype == "circle":
            self._gen_circle()
        elif roomtype == "donut":
            self._gen_donut()
        elif roomtype == "void":
            self._gen_void()
        else:
            raise TypeError("No such valid map type")
        self._place_goal()


    def _gen_circle(self):
        """
        Generate Empty Circle at Center [x, y] of radius r
        param = [[x, y], r]
        """

        center = self.param[0]
        radius = self.param[1]
        for yi in range(self.y_n):
            for xi in range(self.x_n):
                # check if cell center is within bounds
                xpos = xi*self.cellsize+self.cellsize/2
                ypos = yi*self.cellsize+self.cellsize/2
                dist = ((xpos-center[0])**2+(ypos-center[1])**2)**0.5
                if dist >= radius:  # Check radius
                    self.grid[yi, xi] = 1
        # Agent class definition
        self.agt = self._create_agt(x0=center[0], y0=center[1])

    def _gen_donut(self):
        """
        Generate Donut of radius between r1 and r2 at center [x, y]
        where r2 > r1
        param = [[x, y], r1, r2]
        """
        center = self.param[0]
        inner_radius = self.param[1]
        outer_radius = self.param[2]
        for yi in range(self.y_n):
            for xi in range(self.x_n):
                # check if cell center is within bounds
                xpos = xi*self.cellsize+self.cellsize/2
                ypos = yi*self.cellsize+self.cellsize/2
                dist = ((xpos-center[0])**2+(ypos-center[1])**2)**0.5
                if dist >= outer_radius or (dist <= inner_radius and dist > 0):  # Check for the inner radius
                    self.grid[yi, xi] = 1
        # Agent class definition
        self.agt = self._create_agt(x0=center[0] + outer_radius - (outer_radius-inner_radius)/2, y0=center[1])

    def _gen_void(self):
        """
        Generate empty room
        param = [[x1, y1]]
        """
        start = self.param[0]
        # Agent class definition
        self.agt = self._create_agt(x0=start[0], y0=start[1])
    
    def _place_goal(self):
        done = False
        while not done:
            xi = int(np.random.rand()*self.x_n)
            yi = int(np.random.rand()*self.y_n)
            if self.grid[yi, xi] == 0:
                self.grid[yi, xi] = 2
                done = True

    def _create_agt(self, x0=0, y0=0, spd=1, theta0=-45):
        # Agent class definition
        agt = agent.turtle.turtle(x0=x0, y0=y0, spd=spd, theta0=theta0)
        return agt

    def draw_map(self):
        plt.figure(0)
        plt.title("Map Layout")
        plt.imshow(self.grid)
        plt.show()

    def validate(self, screen):
        xi = self.agt.x
        yi = self.agt.y
        r = self.cellsize
        inc = r*math.cos(math.radians(45))

        collider = [
            [yi, xi],
            [yi+r, xi],
            [yi+inc, xi+inc],
            [yi, xi+r],
            [yi-inc, xi+inc],
            [yi-r, xi],
            [yi-inc, xi-inc],
            [yi, xi-r],
            [yi+inc, xi-inc],
        ]

        collisions = [raytools.getCollision(self.cellsize, self.grid, xi, yi) for [yi, xi] in collider]
        for yi, xi in collider:
            pygame.draw.circle(screen, [255, 255, 255], [xi, yi], 2) # debug for collision

        if 2 in collisions: return 2
        elif 1 in collisions: return 1
        else: return 0