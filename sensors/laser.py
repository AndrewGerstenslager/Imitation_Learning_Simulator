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
        collisiondist, collisionpos, collision = self._castx(x, y, theta, screen)
        return collisiondist, collisionpos, collision

    def _castx(self, x, y, theta, screen):
        theta = tool.wrapTheta(theta + self.angle_off)
        rad = math.radians(theta)

        cellidx = tool.getGridIndex(self.cell_size, x, y)
        cellpos = tool.getCellPos(self.cell_size, x, y)
        print(cellidx)

        # x direction
        distx = self.max_range
        if theta < 90 or theta > 270:
            x1 = self.cell_size-cellpos[0] # else, just cellpos 
            y1 = math.tan(rad)*x1 # negative for looking +y
            dist1 = tool.norm(x1,y1)
            collisionpos = [x+math.cos(rad)*self.max_range, y+math.sin(rad)*self.max_range]
            if dist1 > self.max_range:
                # initial check is too far
                distx = self.max_range
                collisiondist = distx # DONE
                collision = False
            else:
                # check first collision
                posx = (cellidx[0]+1)*self.cell_size
                posy = y+y1
                pygame.draw.circle(screen, [255, 0, 0], [posx, posy], 2)
                if tool.getCollision(self.cell_size, self.grid, posx, posy):
                    distx = tool.norm(posx, posy) # return collision dist
                    collisiondist = distx
                    collisionpos = [posx, posy] # DONE
                    collision = True
                else:
                    distx = tool.norm(posx-x, posy-y)
                    yinc = math.tan(rad)*self.cell_size
                    distinc = tool.norm(yinc, self.cell_size)
                    distx += distinc
                    while distx < self.max_range:
                        posx += self.cell_size
                        posy += yinc
                        pygame.draw.circle(screen, [255, 0, 0], [posx, posy], 2)
                        if tool.getCollision(self.cell_size, self.grid, posx, posy):
                            collisionpos = [posx, posy]
                            collision = True
                            break
                        distx += distinc
                    if distx > self.max_range:
                        distx = self.max_range
                        collision = False
                    collisiondist = distx # DONE
        if theta > 90 and theta < 270:
            x1 = cellpos[0] # else, just cellpos 
            y1 = math.tan(rad)*x1 # negative for looking +y 
            dist1 = tool.norm(x1,y1)
            collisionpos = [x+math.cos(rad)*self.max_range, y+math.sin(rad)*self.max_range] # same
            if dist1 > self.max_range:
                # initial check is too far
                distx = self.max_range
                collisiondist = distx # DONE
                collision = False
            else:
                # check first collision
                posx = (cellidx[0])*self.cell_size - 0.01 # THIS COULD CAUSE A BUG LOOKING LEFT 
                posy = y-y1 #flipped sign
                pygame.draw.circle(screen, [255, 0, 0], [posx, posy], 2)
                if tool.getCollision(self.cell_size, self.grid, posx, posy):
                    distx = tool.norm(posx, posy) # return collision dist
                    collisiondist = distx
                    collisionpos = [posx, posy] # DONE
                    collision = True
                else:
                    distx = tool.norm(posx-x, posy-y)
                    yinc = math.tan(rad)*self.cell_size
                    distinc = tool.norm(yinc, self.cell_size)
                    distx += distinc
                    while distx < self.max_range:
                        posx -= self.cell_size # -
                        posy -= yinc         # -
                        pygame.draw.circle(screen, [255, 0, 0], [posx, posy], 2)
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
            collisionpos = [x+math.cos(rad)*self.max_range, y+math.sin(rad)*self.max_range]
            collision = False
        pygame.draw.line(screen, [255, 0, 0], [x, y], collisionpos, width=1)
        return collisiondist, collisionpos, collision