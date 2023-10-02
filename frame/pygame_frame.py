import pygame

def init_frame(WIDTH, HEIGHT):
    # Initialize pygame
    pygame.init()

    # Constants
    WIDTH = 800 + 64
    HEIGHT = 600

    # Set up display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sim")

    return screen

def step(view):
    # Colors
    WHITE = (255, 255, 255)
    GRAY = (80, 80, 80)
    RED = (255, 0, 0)
    view.fill(GRAY)