import pygame
import sys
import math
import time
import numpy as np

sys.path.append('..')
import models.PoseNet
import models.cheatCNN
import torch
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
#envParam = [[800/2, 600/2], inner_radius, outer_radius]
envParam = [width/2, height/2, 0]
map = env.envGenerator.Env(param=envParam,
                           roomtype="datagen",
                           cellsize=cellsize, 
                           width=width, 
                           height=height,
                           seed=seed)

# Agent
agt = map.agt

#map.draw_map()
view.show_map(map.grid)

# CNN
Model = models.PoseNet.NN()
Model.load_state_dict(torch.load("models/saved/Pose_Net_LR0.001_Ep1000_Opt-SGD_LossMSE.pt", map_location=torch.device('cpu')))
Model.eval()


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
    
    i = 0
    clock = pygame.time.Clock()
    recommended = [0, 0, 0]
    while True:
        i += 1
        # exit condition
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # update movement
        keys = pygame.key.get_pressed()
        agt.handle_movement(keys, recommended)
        recorder.step(keys, lasers, agt, view)

        # Drawing env
        view.step()
        off, norm = view.disp_angleoff(agt, map)
        agt.draw(view.screen) 

        # get camera feed TODO: offload
        img = camera.snap(agt)
        if i / 1 >= 1:
            #y_hat = Model(torch.from_numpy(np.rot90(np.moveaxis(img, 2, 0), axes=(1, 2))/255).type(torch.float).view(1, 3, 64, 64))
            #view.disp_pred(float(y_hat)*120-60)
            if recommended == [1, 0, 0]:
               recommended = [1, 0, 0]
            else:
                recommended = models.cheatCNN.readImg(img)
            i = 0


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
