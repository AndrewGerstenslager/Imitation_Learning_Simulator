import pygame
import sys
import math
import time
import numpy as np

sys.path.append('..')
import agent.turtle
import frame.pygame_frame
import env.envGenerator
import sensors.laser
import record.control_record
from sensors.virtualcamera import virtualcamera

cellsize = 10
width = 240
height = 240
camera_res = 64
data_num = 2000
labels = np.zeros((data_num,3))

# Pygame window management
view = frame.pygame_frame.Frame(WIDTH=width, HEIGHT=height, sidebar=camera_res*3)

for i in range(data_num):
    # Env class definitiond
    envParam = [np.random.random()*(width-40)+20, np.random.random()*(height-40)+20, np.random.random()*360]
    map = env.envGenerator.Env(param=envParam,
                            roomtype="datagen",
                            cellsize=cellsize, 
                            width=width, 
                            height=height)
    # Virtual Camera Definition
    laser_max_range = 124
    FOV = 120
    camera = virtualcamera(FOV=FOV, 
                        width=camera_res, 
                        height=camera_res,
                        frame_width=width, 
                        max_range=laser_max_range,
                        view=view,
                        cellsize=cellsize,
                        grid=map.grid)

    # Agent
    agt = map.agt

    #map.draw_map()
    view.show_map(map.grid)

    # Drawing env
    view.step()
    off, norm = view.disp_angleoff(agt, map)
    agt.draw(view.screen) 

    # Take picture
    camera.snap(agt, savesnap=True, snapname="img"+str.zfill(str(i),4))

    pygame.display.flip()

    labels[i, :] = np.array([i, off, norm])

np.savetxt("cache/dataset_index.csv", labels.astype('float'), fmt=["%5i", "%10.5f", "%10.5f"], header='index, angle_offset, distance_to_goal (max_range='+str(laser_max_range)+', FOV='+str(FOV)+')')