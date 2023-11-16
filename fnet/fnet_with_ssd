import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pygame
import torchvision


# Neural network definition
class FNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
# Define ssd
ssd_model = torchvision.models.detection.ssd300(pretrained=True)
ssd_model.eval()  # Set the model to evaluation mode

# Function to get laser readings and convert them to PyTorch tensor
def get_laser_readings(lasers, agent):
    readings = []
    goal_detected = False
    for laser in lasers:
        # Assume the 'cast' function returns a distance
        dist, _, _ = laser.cast(agent.x, agent.y, agent.theta, None, False)
        readings.append(dist)
        if dist < goal_threshold:  # If the goal is closer than this threshold
            goal_detected = True
    return torch.FloatTensor(readings), goal_detected 

def get_object_detections(lasers, agent, ssd_model):
    # Preprocess laser data and convert to tensor (assuming lasers is an image-like input)
    laser_tensor = torch.FloatTensor(lasers)

    # Perform object detection using SSD model
    with torch.no_grad():
        detections = ssd_model(laser_tensor.unsqueeze(0))  # Batch size of 1

    # Extract bounding boxes and labels from detections
    boxes = detections[0]['boxes']
    labels = detections[0]['labels']

    return boxes, labels

# Assume n_lasers is the number of your laser sensors
input_size = n_lasers
hidden_size = 64  # You can tune this
output_size = 3  # Number of actions, for example: turn left, turn right, move forward

# Instantiate the network
net = FNet(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Function to convert laser readings to network input
def laser_to_tensor(lasers):
    # Assuming lasers is a list of distance readings
    return torch.FloatTensor(lasers)

# Main training loop
def train_network(training_data, num_epochs, print_every):
    net.train()  # Set the network to training mode
    for epoch in range(num_epochs):
        for data in training_data:
             # Prepare data
            inputs = get_laser_readings(data['lasers'], data['agent'])
            target_actions = torch.LongTensor(data['actions'])

            optimizer.zero_grad()

             # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, target_actions)
            loss.backward()
            optimizer.step()

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # Print statistics
        if epoch % print_every == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    for epoch in range(num_epochs):
        # ... rest of your loop ...
        # Switch from exploration to exploitation when the goal is detected
        _, goal_detected = get_laser_readings(lasers, agent)
        if goal_detected:
            explore = False
        action = choose_action(lasers, agent, explore)
        execute_action(action, agent)

# Example use in the game loop
def choose_action(lasers, agent, ssd_model, explore):
    with torch.no_grad():  # No need to track gradients here
        boxes, labels = get_object_detections(lasers, agent, ssd_model)
        inputs, goal_detected = get_laser_readings(lasers, agent)
        if explore and not goal_detected:
            # Exploration - random action
            return np.random.choice([0, 1, 2])
        else:
        outputs = net(inputs)
        _, predicted_action = torch.max(outputs, 0)
        return predicted_action.item()

# In your game loop, you would use the network to decide on actions
# Example (pseudo-code):
laser_readings = get_laser_readings()  # Function to get current laser readings
inputs = laser_to_tensor(laser_readings)
outputs = net(inputs)
predicted_actions = outputs.argmax()  # Decide the action based on the highest output value
perform_action(predicted_actions)  # Function to perform the decided action

def execute_action(action, agent):
    # Define the mapping from action to agent's control
    if action == 0:  # Turn left
        agent.theta -= 5  # Assuming theta is in degrees
    elif action == 1:  # Move forward
        agent.x += agent.speed * np.cos(np.radians(agent.theta))
        agent.y += agent.speed * np.sin(np.radians(agent.theta))
    elif action == 2:  # Turn right
        agent.theta += 5
    # Make sure theta stays in the range [0, 360)
    agent.theta = agent.theta % 360

def main():

    train_network(training_data, net, criterion, optimizer, num_epochs, print_every)  # Train the network before starting the main loop

    clock = pygame.time.Clock()

    explore = True  # Start with exploration

    while True:
        # exit condition
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Get laser readings as input for the neural network
        current_laser_readings = get_laser_readings(lasers, agent)

        # Decide whether to explore or exploit based on goal detection
        if goal_detected:
            explore = False

        # Use the neural network to decide on an action
        action = choose_action(lasers, agent, explore)

        # Execute the chosen action
        execute_action(action, agent)

        # Drawing env
        view.step()
        agent.draw(view.screen) 
        pygame.draw.circle(view.screen, [255, 255, 255], [agent.x, agent.y], 2) # debug for collision

        # Check for collision or goal state
        status = map.validate(agent)
        if status == 1:
            print("Crash")
            # Implement what should happen if the agent crashes
            break  # or start a new episode
        elif status == 2:
            print("Goal Reached")
            # Implement what should happen if the agent reaches the goal
            break  # or start a new episode
        
        # Update the screen
        pygame.display.flip()
        # Tick the clock
        clock.tick(60)  # Maintains 60 frames per second

if __name__ == "__main__":
    main()
