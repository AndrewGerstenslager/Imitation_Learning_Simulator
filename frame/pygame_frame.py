import pygame
import matplotlib.pyplot as plt

class Frame():
    def __init__(self, WIDTH, HEIGHT, sidebar):
        # Initialize pygame
        pygame.init()

        # Constants
        self.WIDTH = 800 + 64
        self.HEIGHT = 600
        self.sidebar = sidebar
        self.map_back = None

        # Set up display
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Sim")

    def step(self):
        # Colors
        WHITE = (255, 255, 255)
        GRAY = (80, 80, 80)
        RED = (255, 0, 0)
        if type(self.map_back) != None:
            self.screen.blit(self.map_back, (0, 0))
        else:
            self.screen.fill(GRAY)
        #self._render_sidebar()

    def show_map(self, grid):
        fig = plt.figure(0)
        fig.set_size_inches(self.WIDTH/100, self.HEIGHT/100)
        plt.imshow(grid)
        plt.axis('off')
        plt.savefig('cache/map.png', bbox_inches='tight', pad_inches=0, dpi=130)
        # create a surface object, image is drawn on it.
        self.map_back = pygame.image.load("cache/map.png").convert()


    # def _render_sidebar(self):
