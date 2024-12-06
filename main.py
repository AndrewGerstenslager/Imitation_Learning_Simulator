import pygame
import sys
from typing import List, Tuple, Optional
import numpy as np

sys.path.append("..")
import models.cheatCNN
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
LASER_MAX_RANGE = 124
NUM_LASERS = 20
LASER_FOV = 180
CAMERA_FOV = 120
OUTER_RADIUS = 250
INNER_RADIUS = 150
HALLWAY_LENGTH = 11  # Range 9-20, higher values reduce map complexity
FPS = 20

# UI Text positions
TEXT_X = SCREEN_WIDTH
TEXT_START_Y = SCREEN_HEIGHT - 250
TEXT_SPACING = 30

# Colors
WHITE = (255, 255, 255)


def initialize_game() -> Tuple[frame.pygame_frame.Frame, env.envGenerator.Env]:
    """Initialize the game environment and view."""
    # Pygame window management
    view = frame.pygame_frame.Frame(
        WIDTH=SCREEN_WIDTH, HEIGHT=SCREEN_HEIGHT, sidebar=CAMERA_RES * 3
    )

    # Initialize environment
    seed = int(np.random.random() * 1000)
    env_params = [SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, 0]
    game_map = env.envGenerator.Env(
        param=env_params,
        roomtype="winding",
        cellsize=CELL_SIZE,
        width=SCREEN_WIDTH,
        height=SCREEN_HEIGHT,
        hallway_length=HALLWAY_LENGTH,
        seed=seed,
    )

    return view, game_map


# Initialize game components
view, map = initialize_game()

pygame.font.init()
font = pygame.font.SysFont("Arial", 24)
# Agent
agt = map.agt

# map.draw_map()
view.show_map(map.grid)

# CNN
# Model = models.PoseNet.NN()
# Model.load_state_dict(torch.load("models/saved/Pose_Net_LR0.001_Ep1000_Opt-SGD_LossMSE.pt", map_location=torch.device('cpu')))
# Model.eval()


def initialize_sensors(
    game_map: env.envGenerator.Env,
) -> Tuple[List[sensors.laser.LaserSensor], virtualcamera]:
    """Initialize laser sensors and virtual camera."""
    # Initialize laser sensors
    offset = np.linspace(-LASER_FOV / 2, LASER_FOV / 2, NUM_LASERS)
    lasers = [
        sensors.laser.LaserSensor(
            angle_off=offset[i],
            max_range=LASER_MAX_RANGE,
            cell_size=CELL_SIZE,
            grid=game_map.grid,
        )
        for i in range(NUM_LASERS)
    ]

    # Initialize virtual camera
    camera = virtualcamera(
        FOV=CAMERA_FOV,
        width=CAMERA_RES,
        height=CAMERA_RES,
        frame_width=SCREEN_WIDTH,
        max_range=LASER_MAX_RANGE,
        view=view,
        cellsize=CELL_SIZE,
        grid=game_map.grid,
    )

    return lasers, camera


# Initialize sensors
lasers, camera = initialize_sensors(map)


# Recorder Definition
recorder = record.control_record.recorder()


def draw_ui_text(screen: pygame.Surface) -> None:
    """Draw UI instructions on the screen."""
    instructions = [
        "Drive: W,A,S,D",
        "Record: R",
        "Teaching: E",
        "Self-Driving: F",
        "Load Model: K",
        "Save Model: L",
    ]

    for i, text in enumerate(instructions):
        text_surface = font.render(text, True, WHITE)
        screen.blit(text_surface, (TEXT_X, TEXT_START_Y + i * TEXT_SPACING))


def handle_game_state(valid: int) -> None:
    """Handle game state changes."""
    if valid == 2:
        print("Goal Reached")
    elif valid == 1:
        print("Crash")


def main() -> None:
    """Main game loop."""
    clock = pygame.time.Clock()
    frame_count = 0

    while True:
        frame_count += 1

        # Handle exit condition
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # update movement
        keys = pygame.key.get_pressed()
        agt.handle_movement(keys)
        # recorder.step(keys, lasers, agt, view)

        # Drawing env
        view.step()
        # off, norm = view.disp_angleoff(agt, map)
        agt.draw(view.screen)

        # get camera feed
        image = camera.snap(agt)

        # Check laser distances
        ranges = []
        for laser in lasers:
            dist, cpos, coll = laser.cast(agt.x, agt.y, agt.theta, view.screen, True)
            ranges.append(dist)
        recorder.step(keys, ranges, agt, view, image)

        # Stop condition
        valid = map.validate(view.screen)
        if valid == 2:
            print("Goal Reached")
            # break
        elif valid == 1:
            print("Crash")
            # break

        # Draw UI and update display
        draw_ui_text(view.screen)
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
