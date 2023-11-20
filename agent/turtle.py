import pygame
import sys
sys.path.append('..')
import math

class turtle():
    def __init__(self, x0, y0, spd, theta0):
        self.CAR_WIDTH = 25
        self.CAR_HEIGHT = 35

        #load sprite
        self.spr = pygame.image.load('agent/spot_sprite.png').convert_alpha()
        self.spr = pygame.transform.scale(self.spr, (self.CAR_WIDTH, self.CAR_HEIGHT))
        self.spr = pygame.transform.rotate(self.spr, 270)

        self.agt_Rect = pygame.Rect(x0, y0, self.CAR_WIDTH, self.CAR_HEIGHT)

        # pos, speed and direction
        self.speed = spd
        self.theta = theta0
        self.x = x0
        self.y = y0

    def draw(self, screen):
        rotated_image = pygame.transform.rotate(self.spr, -self.theta)  # Pygame rotates counter-clockwise by default
        new_rect = rotated_image.get_rect(center=self.agt_Rect.center)
        actual_pos = [0, 0]
        actual_pos[0] = new_rect.topleft[0] - self.CAR_WIDTH/2
        actual_pos[1] = new_rect.topleft[1] - self.CAR_HEIGHT/2
        screen.blit(rotated_image, actual_pos)

    def handle_movement(self, keys, recommended=[0, 0, 0]):
        theta = self.theta
        speed = self.speed
        if keys[pygame.K_w] or recommended[0]:
            self.x += speed * math.cos(math.radians(theta))
            self.y += speed * math.sin(math.radians(theta))
        if keys[pygame.K_s] :
            self.x -= speed * math.cos(math.radians(theta))
            self.y -= speed * math.sin(math.radians(theta))
        if keys[pygame.K_a] or recommended[2]:
            theta -= 5  # Decrease for counter-clockwise rotation
        if keys[pygame.K_d] or recommended[1]:
            theta += 5  # Increase for clockwise rotation
        self.agt_Rect.x = self.x
        self.agt_Rect.y = self.y
        self.theta = theta

    def self_drive(self, action_vector):
        theta = self.theta
        speed = self.speed
        if action_vector[0]: #W
            self.x += speed * math.cos(math.radians(theta))
            self.y += speed * math.sin(math.radians(theta))
        if action_vector[2]: #S
            self.x -= speed * math.cos(math.radians(theta))
            self.y -= speed * math.sin(math.radians(theta))
        if action_vector[1]: #A
            theta -= 5  # Decrease for counter-clockwise rotation
        if action_vector[3]: #D
            theta += 5  # Increase for clockwise rotation
        self.agt_Rect.x = self.x
        self.agt_Rect.y = self.y
        self.theta = theta