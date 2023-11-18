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
seed = 40   #seeds with good maps: 40, 2  (keep exploring)
outer_radius = 250
inner_radius = 150
# Env class definition
envParam = [[800/2, 600/2], inner_radius, outer_radius]
map = env.envGenerator.Env(param=envParam,
                           roomtype="course",
                           cellsize=cellsize, 
                           width=width, 
                           height=height,
                           seed=seed)

pygame.font.init()
font = pygame.font.SysFont('Arial', 24)
# Agent
agt = map.agt

#map.draw_map()
view.show_map(map.grid)

# Laser Ranger Definition
lasers = []
laser_max_range = 124
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
recorder = record.control_record.recorder()


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
        

        # Drawing env
        view.step()
        off, norm = view.disp_angleoff(agt, map)
        agt.draw(view.screen) 

        # get camera feed
        image = camera.snap(agt)

        # Check laser distances
        ranges = []
        for laser in lasers:
            dist, cpos, coll = laser.cast(agt.x, agt.y, agt.theta, view.screen, True)
            ranges.append(dist)
        recorder.step(keys, ranges, agt, view, image)

        # Stop condition
        valid = map.validate(view.screen)
        if valid == 2:
            print("Goal Reached")
            #break
        elif valid == 1:
           print("Crash")
           #break
        
        # Blit the text surface onto the screen
        view.screen.blit(font.render('Drive: W,A,S,D', True, (255, 255, 255)), (width, height - 250))
        #view.screen.blit(font.render('Record: R', True, (255, 255, 255)), (width, height - 220))
        view.screen.blit(font.render('Teaching: E', True, (255, 255, 255)), (width, height - 190))
        view.screen.blit(font.render('Self-Driving: F', True, (255, 255, 255)), (width, height - 160))
        view.screen.blit(font.render('Load Model: K', True, (255, 255, 255)), (width, height - 100))
        view.screen.blit(font.render('Save Model: L', True, (255, 255, 255)), (width, height - 70))


        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()
