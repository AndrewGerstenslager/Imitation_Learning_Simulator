import pygame
import sys
import math

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
laser1 = sensors.laser.LaserSensor(0, width/6, cellsize, map.grid)


"""
def handle_keydown_event(event):
    if event.key == pygame.K_i:
        perform_action()3
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
        
        # collect keypress and update control 
        keys = pygame.key.get_pressed()
        agt.handle_movement(keys)

        # Drawing
        view.step()
        agt.draw(view.screen)
        dist, cpos, coll = laser1.cast(agt.x, agt.y, agt.theta, view.screen)
        print(coll)

        # Stop condition
        if map.validate(agt) == 1:
            print("Crash")
            break
        elif map.validate(agt) == 2:
            print("Goal Reached")
            break

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
