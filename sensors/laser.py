import numpy as np
import math
import tools.raytools as tool
import pygame

class LaserSensor():
    """
    Sensor class for laser ranging device. Should intake pose of agent and environment, 
    and return distance of first collision
    """
    def __init__(self, angle_off, max_range, cell_size, grid):
        self.angle_off = angle_off
        self.max_range = max_range
        self.cell_size = cell_size
        self.grid = grid
        self.source = [0.0, 0.0]
        self.hit    = [0.0, 0.0]

    def cast(self, x, y, theta, screen):
        theta = tool.wrapTheta(theta + self.angle_off)
        rad = math.radians(theta)

        cellidx = tool.getGridIndex(self.cell_size, x, y)
        cellpos = tool.getCellPos(self.cell_size, x, y)
        print(cellidx)

        # x direction
        distx = self.max_range
        if theta < 90 or theta > 270:
            x1 = self.cell_size-cellpos[0]
            y1 = math.tan(rad)*x1
            dist1 = tool.norm(x1,y1)
            collisionpos = [x, y]
            if dist1 > self.max_range:
                # initial check is too far
                distx = self.max_range
                collisiondist = distx # DONE
                collision = False
            else:
                # check first collision
                posx = (cellidx[0]+1)*self.cell_size
                posy = y+y1
                if tool.getCollision(self.cell_size, self.grid, posx, posy):
                    distx = tool.norm(posx, posy) # return collision dist
                    collisiondist = distx
                    collisionpos = [posx, posy] # DONE
                    collision = True
                else:
                    distx = tool.norm(posx, posy)
                    yinc = math.tan(rad)*self.cell_size
                    distinc = tool.norm(yinc, self.cell_size)
                    distx += distinc
                    while distx < self.max_range:
                        posx += self.cell_size
                        posy += yinc
                        if tool.getCollision(self.cell_size, self.grid, posx, posy):
                            collisionpos = [posx, posy]
                            collision = True
                            break
                        distx += distinc
                    if distx > self.max_range:
                        distx = self.max_range
                        collision = False
                    collisiondist = distx # DONE
        else:
            collisiondist = self.max_range
            collisionpos = [x, y]
            collision = False
        pygame.draw.line(screen, [255, 0, 0], [x, y], collisionpos, width=1)
        return collisiondist, collisionpos, collision