import pygame
import sys
import math

sys.path.append('..')

import agent.turtle
import frame.pygame_frame

# Pygame window management
view = frame.pygame_frame.init_frame(WIDTH=800+64, HEIGHT=600)
# 
agt = agent.turtle.turtle(x0=0, y0=0, spd=1, theta0=0)

"""
def handle_keydown_event(event):
    if event.key == pygame.K_i:
        perform_action()

def perform_action():
    # Saving screenshot
    #pygame.image.save(screen1, "screenshot.png")
    #print("Screenshot saved!")

    # Checking and printing WASD keys
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        print("W key pressed")
    if keys[pygame.K_a]:
        print("A key pressed")
    if keys[pygame.K_s]:
        print("S key pressed")
    if keys[pygame.K_d]:
        print("D key pressed")
"""

def main():
    clock = pygame.time.Clock()


    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # elif event.type == pygame.KEYDOWN:
            #     handle_keydown_event(event)

        keys = pygame.key.get_pressed()
        agt.handle_movement(keys)

        # Drawing
        frame.pygame_frame.step(view)
        agt.draw(view)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
