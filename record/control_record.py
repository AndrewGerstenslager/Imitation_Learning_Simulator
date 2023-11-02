import pygame
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

class recorder():
    def __init__(self) -> None:
        self.recorded_data = pd.DataFrame()
        self.frame_buffer_input = 0
        self.model = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(4, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.recording = False
        self.self_driving = False
        self.teaching = False
        self.print_prediction = False

    def step(self, keys, rangers, agt, view):

        # Toggle recording with "R" key
        if keys[pygame.K_r] and self.frame_buffer_input == 0:
            self.recording = not(self.recording)
            self.frame_buffer_input += 1
            if not self.recording:
                self.recorded_data.to_csv(f'cache/data_log_test.csv', index=False)
        
        # Toggle self-driving mode with "F" key
        if keys[pygame.K_e] and self.frame_buffer_input == 0:
            self.teaching = not(self.teaching)
            self.frame_buffer_input += 1

        # Toggle self-driving mode with "F" key
        if keys[pygame.K_f] and self.frame_buffer_input == 0:
            self.self_driving = not(self.self_driving)
            self.frame_buffer_input += 1

        
        # Toggle self-driving mode with "F" key
        if keys[pygame.K_c] and self.frame_buffer_input == 0:
            self.print_prediction = not(self.print_prediction)
            self.frame_buffer_input += 1

        if self.teaching:
            input_data = np.array(ranges).reshape(1, -1) / 133
            #true_action_index = np.argmax(np.array([int(keys[pygame.K_w]), int(keys[pygame.K_a]), int(keys[pygame.K_s]), int(keys[pygame.K_d])]))
            action_truth = None#np.array([int(keys[pygame.K_w]), int(keys[pygame.K_a]), int(keys[pygame.K_s]), int(keys[pygame.K_d])]).reshape(1, -1)
            if int(keys[pygame.K_a]):
                action_truth = np.array([0,1,0,0]).reshape(1, -1)
                self.model.fit(input_data, action_truth, epochs=100, validation_data=None, batch_size=1)
            elif int(keys[pygame.K_d]):
                action_truth = np.array([0,0,0,1]).reshape(1, -1)
                self.model.fit(input_data, action_truth, epochs=100, validation_data=None, batch_size=1)
            else:
                action_truth = np.array([1,0,0,0]).reshape(1, -1)
                self.model.fit(input_data, action_truth, epochs=1, validation_data=None, batch_size=1)

            self.model.fit(input_data, action_truth, epochs=1, validation_data=None, batch_size=1)

        if self.print_prediction:
            input_data = np.array(ranges).reshape(1, -1) / 133
            predicted_action = self.model.predict(input_data)
            action_idx = np.argmax(predicted_action)
            action_vector = np.eye(4)[action_idx]
            print(action_vector)

        if self.self_driving:
            input_data = np.array(ranges).reshape(1, a-1) / 133
            predicted_action = self.model.predict(input_data)
            action_idx = np.argmax(predicted_action)
            action_vector = np.eye(4)[action_idx]
            print(action_vector)
            agt.self_drive(action_vector)


        #after everything occurs record data if toggled on
        if self.recording:
            input_data = pd.DataFrame([int(keys[pygame.K_w]), int(keys[pygame.K_a]), int(keys[pygame.K_s]), int(keys[pygame.K_d]), ranges],
                                   index=['W', 'A', 'S', 'D', 'ranges'])
            recorded_data = pd.concat([recorded_data, input_data.T], ignore_index=True)

            print(recorded_data)

        if self.frame_buffer_input > 0:
            self.frame_buffer_input += 1
        if self.frame_buffer_input == 10:
            self.frame_buffer_input = 0