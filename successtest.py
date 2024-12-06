import pygame
import sys
import numpy as np
from typing import List, Tuple, Optional

sys.path.append("..")
import agent.turtle
import frame.pygame_frame
import env.envGenerator
import sensors.laser
import record.control_record
from sensors.virtualcamera import virtualcamera

# Game constants
CELL_SIZE = 10
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CAMERA_RES = 64
MAX_STEPS = 2000
LASER_MAX_RANGE = 124
NUM_LASERS = 20
LASER_FOV = 180
CAMERA_FOV = 120
OUTER_RADIUS = 250
INNER_RADIUS = 150
FPS = 1200

# Pygame window management
view = frame.pygame_frame.Frame(
    WIDTH=SCREEN_WIDTH, HEIGHT=SCREEN_HEIGHT, sidebar=CAMERA_RES * 3
)


def initialize_environment() -> env.envGenerator.Env:
    """Initialize the game environment with random seed."""
    seed = int(np.random.random() * 1000)
    env_params = [[SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2], INNER_RADIUS, OUTER_RADIUS]
    return env.envGenerator.Env(
        param=env_params,
        roomtype="winding",
        cellsize=CELL_SIZE,
        width=SCREEN_WIDTH,
        height=SCREEN_HEIGHT,
        hallway_length=11,
        seed=seed,
    )


def main() -> bool:
    """Main game loop. Returns True if goal is reached, False otherwise."""
    clock = pygame.time.Clock()

    # Initialize environment and map
    map = initialize_environment()

    pygame.font.init()
    font = pygame.font.SysFont("Arial", 24)
    # Agent
    agt = map.agt

    # map.draw_map()
    view.show_map(map.grid)

    # Initialize sensors
    offset = np.linspace(-LASER_FOV / 2, LASER_FOV / 2, NUM_LASERS)
    lasers = [
        sensors.laser.LaserSensor(
            angle_off=offset[i],
            max_range=LASER_MAX_RANGE,
            cell_size=CELL_SIZE,
            grid=map.grid,
        )
        for i in range(NUM_LASERS)
    ]

    camera = virtualcamera(
        FOV=CAMERA_FOV,
        width=CAMERA_RES,
        height=CAMERA_RES,
        frame_width=SCREEN_WIDTH,
        max_range=LASER_MAX_RANGE,
        view=view,
        cellsize=CELL_SIZE,
        grid=map.grid,
    )

    # Recorder Definition
    recorder = record.control_record.recorder("model_20231120_010859.h5")
    recorder.self_driving = True

    i = 0
    ctrl_prev = [0, 0, 0, 0]
    while True:
        i = i + 1
        # exit condition
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # update movement
        keys = pygame.key.get_pressed()

        # Drawing env
        view.step()
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
        elif (
            ctrl_prev[0] == 1
            and ctrl_prev[1] == -1
            and ctrl_prev[2] == 1
            and ctrl_prev[3] == -1
        ):
            agt.self_drive([1, 0, 0])
            # print("infinite loop detected")
            # goal_reached = False
            # break
        elif i >= MAX_STEPS:
            print("Max Time Reached")
            goal_reached = False
            break

        # Display step counter
        step_text = f"{i}/{MAX_STEPS}"
        text_surface = font.render(step_text, True, (255, 255, 255), (0, 0, 0))
        view.screen.blit(text_surface, (SCREEN_WIDTH, SCREEN_HEIGHT - 70))

        pygame.display.flip()
        clock.tick(FPS)
    return goal_reached


if __name__ == "__main__":
    EPISODE_COUNT = 5
    results = [main() for _ in range(EPISODE_COUNT)]
    success_rate = sum(results) / len(results) * 100
    print(f"Results: {results}")
    print(f"Success rate: {success_rate:.1f}%")
