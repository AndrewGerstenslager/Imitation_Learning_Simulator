import pygame
import sys
import math
import numpy as np
import pandas as pd
sys.path.append('..')

import agent.turtle
import frame.pygame_frame
import env.envGenerator
import sensors.laser

cellsize = 10
width = 800
height = 600
camera_res = 64
# Pygame window management
view = frame.pygame_frame.Frame(WIDTH=width+camera_res, HEIGHT=height, sidebar=camera_res)
# Agent class definition
agt = agent.turtle.turtle(x0=width/2, y0=height/2, spd=1, theta0=0)
# Env class definition
map = env.envGenerator.Env(center=[800/2, 600/2], radius=250, cellsize=cellsize, width=width, height=height)
#map.draw_map()
view.show_map(map.grid)

# Laser Ranger Definition
lasers = []
n_lasers = 20
laser_FOV = 180
offset = np.linspace(-laser_FOV/2, laser_FOV/2, n_lasers)
for i in range(n_lasers):
    laser = sensors.laser.LaserSensor(offset[i], width/6, cellsize, map.grid)
    lasers.append(laser)


def main():
    clock = pygame.time.Clock()
    recording = False
    recorded_data = pd.DataFrame()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # elif event.type == pygame.KEYDOWN:
            #     handle_keydown_event(event)
        
        # collect keypress and update control 
        keys = pygame.key.get_pressed()
        agt.handle_movement(keys)

        #check for switch in recording status
        if keys[pygame.K_r]:
            recording = not(recording)
        print(f'recording = {recording}')

        # Drawing
        view.step()
        agt.draw(view.screen)
        pygame.draw.circle(view.screen, [255, 255, 255], [agt.x, agt.y], 2)

        # Check laser distances
        ranges = []
        for i in range(n_lasers):
            laser = lasers[i]
            dist, cpos, coll = laser.cast(agt.x, agt.y, agt.theta, view.screen, False)
            ranges.append(dist)
        #print(ranges)

        # Stop condition
        if map.validate(agt) == 1:
            print("Crash")
            break
        elif map.validate(agt) == 2:
            print("Goal Reached")
            break
        
        #after everything occurs record data if toggled on
        if recording:
            input_data = pd.DataFrame([int(keys[pygame.K_w]), int(keys[pygame.K_a]), int(keys[pygame.K_s]), int(keys[pygame.K_d]), ranges],
                                   index=['W', 'A', 'S', 'D', 'ranges'])
            recorded_data = pd.concat([recorded_data, input_data.T], ignore_index=True)

            print(recorded_data)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
