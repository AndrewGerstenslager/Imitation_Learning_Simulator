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



def main():
    clock = pygame.time.Clock()
    end_steps = 2500

    seed = int(np.random.random()*1000)   #seeds with good maps: 40, 2  (keep exploring)
    outer_radius = 250
    inner_radius = 150
    # Env class definition
    envParam = [[800/2, 600/2], inner_radius, outer_radius]
    map = env.envGenerator.Env(param=envParam,
                            roomtype="random",
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
    recorder = record.control_record.recorder("model_20231120_010859.h5")
    recorder.self_driving = True

    i = 0
    ctrl_prev = [0, 0, 0, 0]
    while True:
        i = i+1
        # exit condition
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # update movement
        keys = pygame.key.get_pressed()

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
        ctrl = recorder.step(keys, ranges, agt, view, image)
        ctrl_prev.pop(0)
        if ctrl[1] == 1:
            ctrl_prev.append(1)
        elif ctrl[2] == 1:
            ctrl_prev.append(-1)
        else:
            ctrl_prev.append(0)


        # Stop condition
        valid = map.validate(view.screen)
        if valid == 2:
            print("Goal Reached")
            goal_reached = True
            break
        elif valid == 1:
            print("Crash")
            goal_reached = False
            break
        elif ctrl_prev[0] == 1 and ctrl_prev[1] == -1 and ctrl_prev[2] == 1 and ctrl_prev[3] == -1:
            agt.self_drive([1,0,0])
            #print("infinite loop detected")
            #goal_reached = False
            #break
        elif i >= end_steps:
            print("Max Time Reached")
            goal_reached = False
            break
        
        # Blit the text surface onto the screen
        view.screen.blit(font.render(str(i)+"/"+str(end_steps), True, (255, 255, 255), (0, 0, 0)), (width, height - 70))


        pygame.display.flip()
        clock.tick(1200)
    return goal_reached

if __name__ == "__main__":
    ep_count = 5
    results = []
    for episode in range(ep_count):
        success = main()
        results.append(success)
    print(results)

