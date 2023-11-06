import numpy as np
import pygame
import math
import matplotlib.pyplot as plt
import sensors.laser


class virtualcamera():
    def __init__(self, FOV, width, height, frame_width, max_range, view, cellsize, grid):
        # Laser Ranger Definition
        self.h = max_range*7
        self.lasers = []
        self.width = width
        self.height = height
        self.frame_width = frame_width
        self.max_range = max_range
        self.view = view
        n_lasers = width
        self.FOV=FOV
        self.floor_shadow = None
        offset = np.linspace(-self.FOV/2, self.FOV/2, n_lasers)
        for i in range(n_lasers):
            laser = sensors.laser.LaserSensor(angle_off=offset[i], 
                                              max_range=self.max_range, 
                                              cell_size=cellsize, 
                                              grid=grid)
            self.lasers.append(laser)
        
        # floor/ceiling shadow map
        min_height = int(round((self.h/self.max_range)/2)*2)
        depth = int(self.height/2 - min_height/2)
        self.shadowmap = np.clip(1-np.arange(0, self.height/2)/depth, 0, 1)
        self.shadowmap = np.concatenate((self.shadowmap, np.flip(self.shadowmap)))
        self.backimg = np.stack((self.shadowmap,)*self.width, axis=0)*100  # gray background
        self.backimg = self.backimg.astype('int')

        
    def snap(self, agt):
        # Check laser distances
        slices = []
        dists = []
        perps = []
        gridtypes = []
        for laser in self.lasers:
            if laser is self.lasers[0] or laser is self.lasers[-1]:
                dist, cpos, coll = laser.cast(agt.x, agt.y, agt.theta, self.view.screen, False, camera_beam=True)
            else: dist, cpos, coll = laser.cast(agt.x, agt.y, agt.theta, self.view.screen, False, camera_beam=False)
            perp_laser = math.cos(math.radians(abs(laser.angle_off)))*dist
            gridtypes.append(coll)
            perps.append(perp_laser)
            dists.append(dist)
            lineheight = int(round((self.h/perp_laser)/2)*2)
            slices.append(lineheight)

        # create image feed
        img = np.stack((self.backimg,)*3, axis=-1)
        for i in range(len(slices)):
            slice = slices[i]
            h1 = int(self.height/2-slice/2)
            h2 = int(self.height/2+slice/2)
            if slice >= self.height:
                h1 = 0
                h2 = self.height
            img[i, h1:h2, :] = 255-255*(dists[i]/self.max_range)
            if gridtypes[i] == 2:
                img[i, h1:h2, 1] = 0
                img[i, h1:h2, 2] = 0
        # plt.imshow(img)
        # plt.show()
        canv = pygame.pixelcopy.make_surface(img.astype(int))
        canv = pygame.transform.scale(canv, (self.width*3, self.height*3))
        # canv = pygame.surfarray.blit_array(self.view.screen, img, dest=[self.width, 0])
        self.view.screen.blit(canv, (self.frame_width, 0))
