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
width = 800
height = 600
camera_res = 64

# Pygame window management
view = frame.pygame_frame.Frame(WIDTH=width, HEIGHT=height, sidebar=camera_res*3)

outer_radius = 250
inner_radius = 150
# Env class definition
envParam = [[800/2, 600/2], inner_radius, outer_radius]
map = env.envGenerator.Env(param=envParam,
                           roomtype="grid",
                           cellsize=cellsize, 
                           width=width, 
                           height=height)

# Agent
agt = map.agt

#map.draw_map()
view.show_map(map.grid)

# Laser Ranger Definition
lasers = []
laser_max_range = width/6
n_lasers = 20
laser_FOV = 180
offset = np.linspace(-laser_FOV/2, laser_FOV/2, n_lasers)
for i in range(n_lasers):
    laser = sensors.laser.LaserSensor(angle_off=offset[i], max_range=laser_max_range, cell_size=cellsize, grid=map.grid)
    lasers.append(laser)

camera = virtualcamera(FOV=120, 
                    width=camera_res, 
                    height=camera_res,
                    frame_width=width, 
                    max_range=laser_max_range,
                    view=view,
                    cellsize=cellsize,
                    grid=map.grid)


# Recorder Definition
# recorder = record.control_record.recorder()


def main():
    clock = pygame.time.Clock()
    while True:
        # exit condition
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # update movement
        keys = pygame.key.get_pressed()
        agt.handle_movement(keys)
        # recorder.step(keys, lasers, agt, view)

        # Drawing env
        view.step()
        agt.draw(view.screen) 

        # get camera feed
        print(pygame.time.get_ticks())
        if pygame.time.get_ticks() > 5000 and pygame.time.get_ticks() < 5500:
            savesnap = True
        else: savesnap = False
        camera.snap(agt, savesnap)

        # Check laser distances
        ranges = []
        for laser in lasers:
            dist, cpos, coll = laser.cast(agt.x, agt.y, agt.theta, view.screen, True)
            ranges.append(dist)
        #print(ranges)

        # Stop condition
        valid = map.validate(view.screen)
        if valid == 2:
            print("Goal Reached")
            break
        elif valid == 1:
           print("Crash")
           break

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
