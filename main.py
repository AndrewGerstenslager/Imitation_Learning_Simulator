import pygame
import sys
import math
import time
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


model = keras.Sequential([
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


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
agt = agent.turtle.turtle(x0=width/2 + outer_radius - (outer_radius-inner_radius)/2, y0=height/2, spd=1, theta0=-90)

# Env class definition
envParam = [[800/2, 600/2], inner_radius, outer_radius]
map = env.envGenerator.Env(param=envParam,
                           roomtype="donut",
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
    self_driving = False
    teaching = False
    print_prediction = False
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
                recorded_data.to_csv(f'data_log_{time.time() * 1000}.csv', index=False)
        
        # Toggle self-driving mode with "F" key
        if keys[pygame.K_e] and frame_buffer_input == 0:
            teaching = not(teaching)
            frame_buffer_input += 1

        # Toggle self-driving mode with "F" key
        if keys[pygame.K_f] and frame_buffer_input == 0:
            self_driving = not(self_driving)
            frame_buffer_input += 1

        # Toggle self-driving mode with "F" key
        if keys[pygame.K_c] and frame_buffer_input == 0:
            print_prediction = not(print_prediction)
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

        if teaching:
            input_data = np.array(ranges).reshape(1, -1) / 133
            #true_action_index = np.argmax(np.array([int(keys[pygame.K_w]), int(keys[pygame.K_a]), int(keys[pygame.K_s]), int(keys[pygame.K_d])]))
            action_truth = None#np.array([int(keys[pygame.K_w]), int(keys[pygame.K_a]), int(keys[pygame.K_s]), int(keys[pygame.K_d])]).reshape(1, -1)
            if int(keys[pygame.K_a]):
                action_truth = np.array([0,1,0,0]).reshape(1, -1)
                model.fit(input_data, action_truth, epochs=100, validation_data=None, batch_size=1)
            elif int(keys[pygame.K_d]):
                action_truth = np.array([0,0,0,1]).reshape(1, -1)
                model.fit(input_data, action_truth, epochs=100, validation_data=None, batch_size=1)
            else:
                action_truth = np.array([1,0,0,0]).reshape(1, -1)
                model.fit(input_data, action_truth, epochs=1, validation_data=None, batch_size=1)

            model.fit(input_data, action_truth, epochs=1, validation_data=None, batch_size=1)

        if print_prediction:
            input_data = np.array(ranges).reshape(1, -1) / 133
            predicted_action = model.predict(input_data)
            action_idx = np.argmax(predicted_action)
            action_vector = np.eye(4)[action_idx]
            print(action_vector)

        if self_driving:
            input_data = np.array(ranges).reshape(1, a-1) / 133
            predicted_action = model.predict(input_data)
            action_idx = np.argmax(predicted_action)
            action_vector = np.eye(4)[action_idx]
            print(action_vector)
            agt.self_drive(action_vector)

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

        if frame_buffer_input > 0:
            frame_buffer_input += 1
        if frame_buffer_input == 10:
            frame_buffer_input = 0
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
