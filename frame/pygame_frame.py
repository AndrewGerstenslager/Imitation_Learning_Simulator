import pygame

class Frame():
    def __init__(self, WIDTH, HEIGHT, sidebar):
        # Initialize pygame
        pygame.init()

        # Constants
        WIDTH = 800 + 64
        HEIGHT = 600
        self.sidebar = sidebar

        # Set up display
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Sim")

    def step(self):
        # Colors
        WHITE = (255, 255, 255)
        GRAY = (80, 80, 80)
        RED = (255, 0, 0)
        self.screen.fill(GRAY)
        #self._render_sidebar()

    # def _render_sidebar(self):
