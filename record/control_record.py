import pygame
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tkinter as tk
from tkinter import filedialog
import datetime
import os
from sklearn.utils import shuffle
from models.robot_model import get_model


'''
self.model = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])
self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
'''


class recorder():
    def __init__(self) -> None:
        self.recorded_data = pd.DataFrame()
        self.frame_buffer_input = 0
        self.model = get_model()
        self.recording = False
        self.self_driving = False
        self.teaching = False
        self.print_prediction = False
        self.oversampling_weight = {'W': 1, 'A': 50, 'D': 50}


    def load_model(self):
        root = tk.Tk()
        root.withdraw()  # Hide the Tkinter root window
        initialdir = os.getcwd()  # Get the current working directory
        filepath = filedialog.askopenfilename(initialdir=initialdir)
        if filepath:
            self.model = tf.keras.models.load_model(filepath)
            print("LOADED MODEL")
        root.destroy()

    def save_model(self):
        root = tk.Tk()
        root.withdraw()  # Hide the Tkinter root window
        initialdir = os.getcwd()  # Get the current working directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_{timestamp}.h5"

        self.model.save(filename)
        print("SAVED MODEL")
        root.destroy()

    def toggle_teaching_mode(self, keys):
        self.teaching = not self.teaching
        print(f"TOGGLED TEACHING TO {self.teaching}")
        if not self.teaching:
            print("TRAINING MODEL")
            # When teaching is toggled off, train the model
            self.train_model_on_recorded_data()
            #self.recorded_data = pd.DataFrame()

    def train_model_on_recorded_data(self):
        if not self.recorded_data.empty:
            # Assuming 'image' and 'lidar' are columns storing the respective data
            image_inputs = np.array(self.recorded_data['image'].tolist())  # Convert list of lists to array
            lidar_inputs = np.array(self.recorded_data['lidar'].tolist())  # Convert list of lists to array
            outputs = np.array(self.recorded_data['action'].tolist())

            # Shuffle the data
            indices = np.arange(image_inputs.shape[0])
            np.random.shuffle(indices)
            image_inputs = image_inputs[indices]
            lidar_inputs = lidar_inputs[indices]
            outputs = outputs[indices]

            # Train the model
            self.model.fit([image_inputs, lidar_inputs], outputs, epochs=1, batch_size=1, validation_data=None)

            # Clear the recorded data
            #self.recorded_data = pd.DataFrame(columns=['image', 'lidar', 'action'])
        else:
            print("DATA EMPTY")

    def step(self, keys, ranges, agt, view, image):
        
        # Load model with "K" key
        if keys[pygame.K_k] and self.frame_buffer_input == 0:
            self.load_model()
            self.frame_buffer_input += 1

        # Save model with "L" key
        if keys[pygame.K_l] and self.frame_buffer_input == 0:
            self.save_model()
            self.frame_buffer_input += 1

        # Toggle teaching mode with "E" key
        if keys[pygame.K_e] and self.frame_buffer_input == 0:
            self.recording = False
            print(f'RECORDING = {self.recording}')
            self.frame_buffer_input += 1
            if not self.recording:
                print(f'RECORDED DATA SHAPE = {self.recorded_data.shape}')
            self.train_model_on_recorded_data()

        # Toggle recording mode with "R" key
        if keys[pygame.K_r] and self.frame_buffer_input == 0:
            self.recording = not(self.recording)
            print(f'RECORDING = {self.recording}')
            self.frame_buffer_input += 1
            if not self.recording:
                print(f'RECORDED DATA SHAPE = {self.recorded_data.shape}')

        # Toggle self-driving mode with "F" key
        if keys[pygame.K_f] and self.frame_buffer_input == 0:
            self.self_driving = not(self.self_driving)
            print(f'SELF DRIVING = {self.self_driving}')
            self.frame_buffer_input += 1

        if self.recording:
            lidar_data = np.array(ranges) / 133 # Normalize input data
            image_data = np.array(image) / 255.0  # Normalize pixel values
            action_truth = None
            num_times = 0

            # Record data with oversampling for 'A' and 'D'
            if int(keys[pygame.K_a]):
                num_times = 1
                action_truth = np.array([0, 1, 0])
            elif int(keys[pygame.K_d]):
                num_times = 1
                action_truth = np.array([0, 0, 1])
            else:
                num_times = 1
                action_truth = np.array([1, 0, 0])

            # Create a DataFrame row for the current frame
            frame_data = {'image': [image_data.tolist()], 'lidar': [lidar_data.tolist()], 'action': [action_truth.tolist()]}
            frame_df = pd.DataFrame(frame_data)
            
            # Append the frame data to the recorded data
            for _ in range(num_times):
                self.recorded_data = pd.concat([self.recorded_data, frame_df], ignore_index=True)
            #print(self.recorded_data)

        if self.self_driving:
            # Ensure lidar_data is a two-dimensional array with shape (1, num_features)
            lidar_data = np.array(ranges).reshape(1, -1) / 133  # Normalize input data
            # Ensure image_data is a four-dimensional array with shape (1, height, width, channels)
            image_data = np.array(image).reshape(1, 64, 64, 3) / 255.0  # Normalize pixel values assuming the image is 64x64x3

            # Use model.predict to get the action probabilities
            # The input should be a list where each element is a batch of inputs for one of the model's inputs
            predicted_action = self.model.predict([image_data, lidar_data])
            action_idx = np.argmax(predicted_action, axis=1)  # Use axis=1 to get the index of the max value in each row
            action_vector = np.eye(3)[action_idx]  # Convert to one-hot encoded actions

            print(action_vector.squeeze())  # Squeeze to remove the batch dimension for printing
            agt.self_drive(action_vector.squeeze())  # Assuming agt.self_drive expects a 1D array as input

        if self.frame_buffer_input > 0:
            self.frame_buffer_input += 1
        if self.frame_buffer_input == 10:
            self.frame_buffer_input = 0