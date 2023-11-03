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

cellsize = 10
width = 800
height = 600
camera_res = 64

# Pygame window management
view = frame.pygame_frame.Frame(WIDTH=width+camera_res, HEIGHT=height, sidebar=camera_res)

outer_radius = 250
inner_radius = 150
# Env class definition
envParam = [[800/2, 600/2], inner_radius, outer_radius]
map = env.envGenerator.Env(param=envParam,
                           roomtype="donut",
                           cellsize=cellsize, 
                           width=width, 
                           height=height)

# Agent
agt = map.agt

#map.draw_map()
view.show_map(map.grid)

# Laser Ranger Definition
lasers = []
n_lasers = 20
laser_FOV = 180
offset = np.linspace(-laser_FOV/2, laser_FOV/2, n_lasers)
for i in range(n_lasers):
    laser = sensors.laser.LaserSensor(angle_off=offset[i], max_range=int(width/6), cell_size=cellsize, grid=map.grid)
    lasers.append(laser)


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
        pygame.draw.circle(view.screen, [255, 255, 255], [agt.x, agt.y], 2) # debug for collision

        # Check laser distances
        ranges = []
        for laser in lasers:
            dist, cpos, coll = laser.cast(agt.x, agt.y, agt.theta, view.screen, False)
            ranges.append(dist)
        #print(ranges)

        # Stop condition # TODO: make collision better
        if map.validate(agt) == 1:
            print("Crash")
            break
        elif map.validate(agt) == 2:
           print("Goal Reached")
           break

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
