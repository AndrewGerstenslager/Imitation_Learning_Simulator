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
    
    def cast(self, x, y, theta, screen, show_beam=True, camera_beam=False, debug=False):
        collisiondistx, collisionposx, collisionx = self._castx(x, y, theta, screen, debug)
        collisiondisty, collisionposy, collisiony = self._casty(x, y, theta, screen, debug)

        if collisiondistx <= collisiondisty:
            collisiondist = collisiondistx
            collisionpos = collisionposx
            collision = collisionx
            
        else:
            collisiondist = collisiondisty
            collisionpos = collisionposy
            collision = collisiony

        if show_beam: pygame.draw.line(screen, [255, 0, 0], [x, y], collisionpos, width=1)
        if camera_beam: pygame.draw.line(screen, [255, 255, 0], [x, y], collisionpos, width=2)


        return collisiondist, collisionpos, collision

    def _castx(self, x, y, theta, screen, debug):
        theta = tool.wrapTheta(theta + self.angle_off)
        rad = math.radians(theta)

        cellidx = tool.getGridIndex(self.cell_size, x, y)
        cellpos = tool.getCellPos(self.cell_size, x, y)
        collisiontype = 0

        # x direction
        distx = self.max_range
        if theta < 90 or theta > 270:
            x1 = self.cell_size-cellpos[0] # else, just cellpos 
            y1 = math.tan(rad)*x1 # negative for looking +y
            dist1 = tool.norm(x1,y1)
            if dist1 > self.max_range:
                # initial check is too far
                collision = False
            else:
                # check first collision
                posx = (cellidx[0]+1)*self.cell_size
                posy = y+y1
                if debug: pygame.draw.circle(screen, [255, 0, 0], [posx, posy], 2)
                if tool.getCollision(self.cell_size, self.grid, posx, posy):
                    distx = tool.norm(posx, posy) # return collision dist
                    collisiondist = distx
                    collisionpos = [posx, posy] # DONE
                    collisiontype = tool.getCollision(self.cell_size, self.grid, posx, posy)
                    if debug: pygame.draw.circle(screen, [255, 0, 255], [posx, posy], 3)
                    collision = True
                else:
                    distx = tool.norm(posx-x, posy-y)
                    yinc = math.tan(rad)*self.cell_size
                    distinc = tool.norm(yinc, self.cell_size)
                    distx += distinc
                    while distx < self.max_range:
                        posx += self.cell_size
                        posy += yinc
                        if debug: pygame.draw.circle(screen, [255, 0, 0], [posx, posy], 2)
                        if tool.getCollision(self.cell_size, self.grid, posx, posy):
                            collisionpos = [posx, posy]
                            collisiondist = distx
                            collisiontype = tool.getCollision(self.cell_size, self.grid, posx, posy)
                            if debug: pygame.draw.circle(screen, [255, 0, 255], [posx, posy], 3)
                            collision = True
                            break
                        distx += distinc
                    if distx >= self.max_range: # must be here
                        collision = False
        elif theta > 90 and theta < 270:
            x1 = cellpos[0] # else, just cellpos 
            y1 = math.tan(rad)*x1 # negative for looking +y 
            dist1 = tool.norm(x1,y1)
            if dist1 > self.max_range:
                # initial check is too far
                collision = False
            else:
                # check first collision
                posx = (cellidx[0])*self.cell_size - 0.01 # THIS COULD CAUSE A BUG LOOKING LEFT 
                posy = y-y1 #flipped sign
                if debug: pygame.draw.circle(screen, [255, 0, 0], [posx, posy], 2)
                if tool.getCollision(self.cell_size, self.grid, posx, posy):
                    distx = tool.norm(posx, posy) # return collision dist
                    collisiondist = distx
                    collisionpos = [posx, posy] # DONE
                    collisiontype = tool.getCollision(self.cell_size, self.grid, posx, posy)
                    if debug: pygame.draw.circle(screen, [255, 0, 255], [posx, posy], 3)
                    collision = True
                else:
                    distx = tool.norm(posx-x, posy-y)
                    yinc = math.tan(rad)*self.cell_size
                    distinc = tool.norm(yinc, self.cell_size)
                    distx += distinc
                    while distx < self.max_range:
                        posx -= self.cell_size # -
                        posy -= yinc         # -
                        if debug: pygame.draw.circle(screen, [255, 0, 0], [posx, posy], 2)
                        if tool.getCollision(self.cell_size, self.grid, posx, posy):
                            collisionpos = [posx, posy]
                            collisiondist = distx
                            collisiontype = tool.getCollision(self.cell_size, self.grid, posx, posy)
                            if debug: pygame.draw.circle(screen, [255, 0, 255], [posx, posy], 3)
                            collision = True
                            break
                        distx += distinc
                    if distx >= self.max_range: # must be here
                        collision = False
        else:
            collision = False
        if collision == False:
            collisionpos = [x+math.cos(rad)*self.max_range, y+math.sin(rad)*self.max_range]
            collisiondist = self.max_range
        
        return collisiondist, collisionpos, collisiontype
    
    def _casty(self, x, y, theta, screen, debug):
        theta = tool.wrapTheta(theta + self.angle_off)
        rad = math.radians(theta)

        cellidx = tool.getGridIndex(self.cell_size, x, y)
        cellpos = tool.getCellPos(self.cell_size, x, y)

        collisiontype = 0

        # y direction
        disty = self.max_range
        if theta < 180 and theta > 0:
            y1 = self.cell_size-cellpos[1] # else, just cellpos 
            x1 = y1/math.tan(rad) # negative for looking +y
            dist1 = tool.norm(x1,y1)
            if dist1 > self.max_range:
                # initial check is too far
                collision = False
            else:
                # check first collision
                posy = (cellidx[1]+1)*self.cell_size
                posx = x+x1
                if debug: pygame.draw.circle(screen, [0, 255, 0], [posx, posy], 2)
                if tool.getCollision(self.cell_size, self.grid, posx, posy):
                    disty = tool.norm(posx, posy) # return collision dist
                    collisiondist = disty
                    collisionpos = [posx, posy] # DONE
                    collisiontype = tool.getCollision(self.cell_size, self.grid, posx, posy)
                    if debug: pygame.draw.circle(screen, [0, 255, 255], [posx, posy], 3)
                    collision = True
                else:
                    disty = tool.norm(posx-x, posy-y)
                    xinc = self.cell_size/math.tan(rad)
                    distinc = tool.norm(xinc, self.cell_size)
                    disty += distinc
                    while disty < self.max_range:
                        posy += self.cell_size
                        posx += xinc
                        if debug: pygame.draw.circle(screen, [0, 255, 0], [posx, posy], 2)
                        if tool.getCollision(self.cell_size, self.grid, posx, posy):
                            collisiondist = disty
                            collisionpos = [posx, posy]
                            collisiontype = tool.getCollision(self.cell_size, self.grid, posx, posy)
                            if debug: pygame.draw.circle(screen, [0, 255, 255], [posx, posy], 3)
                            collision = True
                            break
                        disty += distinc
                    if disty >= self.max_range: # must be here
                        collision = False
        elif theta > 180:
            y1 = cellpos[1] # else, just cellpos 
            x1 = -y1/math.tan(rad) # negative for looking +y
            dist1 = tool.norm(x1,y1)
            if dist1 > self.max_range:
                # initial check is too far
                collision = False
            else:
                # check first collision
                posy = (cellidx[1])*self.cell_size - 0.01 # could cause a bug looking down
                posx = x+x1
                if debug: pygame.draw.circle(screen, [0, 255, 0], [posx, posy], 2)
                if tool.getCollision(self.cell_size, self.grid, posx, posy):
                    disty = tool.norm(posx, posy) # return collision dist
                    collisiondist = disty
                    collisionpos = [posx, posy] # DONE
                    collisiontype = tool.getCollision(self.cell_size, self.grid, posx, posy)
                    if debug: pygame.draw.circle(screen, [0, 255, 255], [posx, posy], 3)
                    collision = True
                else:
                    disty = tool.norm(posx-x, posy-y)
                    xinc = -self.cell_size/math.tan(rad)
                    distinc = tool.norm(xinc, self.cell_size)
                    disty += distinc
                    while disty < self.max_range:
                        posy -= self.cell_size
                        posx += xinc
                        if debug: pygame.draw.circle(screen, [0, 255, 0], [posx, posy], 2)
                        if tool.getCollision(self.cell_size, self.grid, posx, posy):
                            collisionpos = [posx, posy]
                            collisiondist = disty
                            collision = True
                            collisiontype = tool.getCollision(self.cell_size, self.grid, posx, posy)
                            if debug: pygame.draw.circle(screen, [0, 255, 255], [posx, posy], 3)
                            break
                        disty += distinc
                    if disty >= self.max_range: # must be here
                        collision = False
        else:
            collision = False
        if collision == False:
            collisionpos = [x+math.cos(rad)*self.max_range, y+math.sin(rad)*self.max_range]
            collisiondist = self.max_range
        return collisiondist, collisionpos, collisiontype