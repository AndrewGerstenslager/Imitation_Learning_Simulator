import numpy as np
import pygame
import math
import sensors.laser


class virtualcamera():
    def __init__(self, FOV, width, height, max_range, view, cellsize, grid):
        # Laser Ranger Definition
        self.h = 10
        self.lasers = []
        self.width = width
        self.height = height
        self.max_range = max_range
        self.view = view
        n_lasers = width
        self.FOV=FOV
        offset = np.linspace(-self.FOV/2, self.FOV/2, n_lasers)
        for i in range(n_lasers):
            laser = sensors.laser.LaserSensor(angle_off=offset[i], 
                                              max_range=self.max_range, 
                                              cell_size=cellsize, 
                                              grid=grid)
            self.lasers.append(laser)
        
    def snap(self, agt):
        # Check laser distances
        slice = []
        for laser in self.lasers:
            dist, cpos, coll = laser.cast(agt.x, agt.y, agt.theta, self.view.screen, False)
            perp_laser = math.cos(math.radians(abs(laser.angle_off)))*dist
            lineheight = int(round((self.h/perp_laser)/2)*2)
            #perpdist.append(perp_laser)

        # create image feed