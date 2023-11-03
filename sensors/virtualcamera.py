import numpy as np
import pygame
import math
import matplotlib.pyplot as plt
import sensors.laser


class virtualcamera():
    def __init__(self, FOV, width, height, frame_width, max_range, view, cellsize, grid):
        # Laser Ranger Definition
        self.h = max_range*10
        self.lasers = []
        self.width = width
        self.height = height
        self.frame_width = frame_width
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
        slices = []
        dists = []
        perps = []
        for laser in self.lasers:
            dist, cpos, coll = laser.cast(agt.x, agt.y, agt.theta, self.view.screen, False)
            perp_laser = math.cos(math.radians(abs(laser.angle_off)))*dist
            perps.append(perp_laser)
            dists.append(dist)
            lineheight = int(round((self.h/perp_laser)/2)*2)
            slices.append(lineheight)

        # create image feed
        img = np.ones([self.width, self.height])*50  # gray background

        for i in range(len(slices)):
            slice = slices[i]
            h1 = int(self.height/2-slice/2)
            h2 = int(self.height/2+slice/2)
            img[i, h1:h2] = 255-255*(dists[i]/self.max_range)
        # plt.imshow(img)
        # plt.show()
        canv = pygame.pixelcopy.make_surface(img.astype(int))
        # canv = pygame.surfarray.blit_array(self.view.screen, img, dest=[self.width, 0])
        self.view.screen.blit(canv, (self.frame_width, 0))
