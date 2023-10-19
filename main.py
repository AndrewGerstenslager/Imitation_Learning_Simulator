import pygame
import sys
import math
import time
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

MODEL_PATH = "model_1.keras"
model = load_model(MODEL_PATH)

sys.path.append('..')
import agent.turtle
import frame.pygame_frame
import env.envGenerator
import sensors.laser

cellsize = 10
width = 800
height = 600
camera_res = 64

outer_radius = 250
inner_radius = 150


# Pygame window management
view = frame.pygame_frame.Frame(WIDTH=width+camera_res, HEIGHT=height, sidebar=camera_res)
# Agent class definition
agt = agent.turtle.turtle(x0=width/2 + inner_radius/2, y0=height/2, spd=1, theta0=0)

# Env class definition
map = env.envGenerator.Env(center=[800/2, 600/2], 
                           outer_radius=outer_radius, 
                           inner_radius=inner_radius, 
                           cellsize=cellsize, 
                           width=width, 
                           height=height)
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


def main():
    clock = pygame.time.Clock()
    recording = False
    self_driving = False  # Flag for self-driving mode
    recorded_data = pd.DataFrame()
    frame_buffer_input = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        #if not self_driving:
        agt.handle_movement(keys)
        
        # Toggle recording with "R" key
        if keys[pygame.K_r] and frame_buffer_input == 0:
            recording = not(recording)
            frame_buffer_input += 1
            if not recording:
                recorded_data.to_csv(f'data_log_{time.time() * 1000}.csv')
        
        # Toggle self-driving mode with "F" key
        if keys[pygame.K_f] and frame_buffer_input == 0:
            self_driving = not(self_driving)
            frame_buffer_input += 1
         

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

        if self_driving:
            input_data = np.array(ranges).reshape(1, -1)
            scaler = MinMaxScaler(feature_range=(0, 133))
            X = scaler.fit_transform(input_data)
            predicted_action = model.predict(X)
            print(predicted_action)
            '''
            action_idx = np.argmax(predicted_action)
            if action_idx == 0:  # W
                agt.move_forward()
            elif action_idx == 1:  # A
                agt.turn_left()
            elif action_idx == 2:  # S
                agt.move_backward()
            elif action_idx == 3:  # D
                agt.turn_right()'''

        # Stop condition
        #if map.validate(agt) == 1:
        #    print("Crash")
        #    break
        #elif map.validate(agt) == 2:
        #    print("Goal Reached")
        #    break
        
        #after everything occurs record data if toggled on
        if recording:
            input_data = pd.DataFrame([int(keys[pygame.K_w]), int(keys[pygame.K_a]), int(keys[pygame.K_s]), int(keys[pygame.K_d]), ranges],
                                   index=['W', 'A', 'S', 'D', 'ranges'])
            recorded_data = pd.concat([recorded_data, input_data.T], ignore_index=True)

            print(recorded_data)

        if frame_buffer_input > 0:
            frame_buffer_input += 1
        if frame_buffer_input == 10:
            frame_buffer_input = 0
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
