import pygame
import numpy as np
import tools.raytools
from PIL import Image
import matplotlib.pyplot as plt

class Frame():
    def __init__(self, WIDTH, HEIGHT, sidebar):
        # Initialize pygame
        pygame.init()

        # Constants
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.sidebar = sidebar
        self.map_back = None

        # Set up display
        self.screen = pygame.display.set_mode((WIDTH+sidebar, HEIGHT))
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

    def disp_angleoff(self, agt, env):
        off = tools.raytools.relativeAngle([agt.x, agt.y], agt.theta, env.goalpos)
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render(str.zfill(str(round(off,2)),6), True, (255, 0, 0),(0,0,0))
        textRect = text.get_rect()
        textRect.center = (self.WIDTH+self.sidebar/2, self.sidebar*1.5)
        self.screen.blit(text, textRect)
        dx = env.goalpos[0]-agt.x
        dy = env.goalpos[1]-agt.y
        norm = tools.raytools.norm(dx, dy)
        return off, norm
    
    def disp_pred(self, pred):
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render(str.zfill(str(round(pred,2)),6), True, (0, 0, 255),(0,0,0))
        textRect = text.get_rect()
        textRect.center = (self.WIDTH+self.sidebar/2, self.sidebar*3)
        self.screen.blit(text, textRect)

    def show_map(self, grid):
        map_img_dir = "cache/map.png"
        grid_1bit = grid/2*255
        grid_img = grid_1bit.astype('uint8')
        im = Image.fromarray(grid_img, mode="L") # could cause a bug if different values are used.
        im.save(map_img_dir)
        # im.show()
        # fig = plt.figure(0)
        # fig.set_size_inches(self.WIDTH/100, self.HEIGHT/100)
        # plt.imshow(grid_img)
        # plt.axis('off')
        # plt.show()
        #plt.savefig('cache/map.png', bbox_inches='tight', pad_inches=0, dpi=130)
        #create a surface object, image is drawn on it.
        self.map_back = pygame.image.load(map_img_dir).convert()
        self.map_back = pygame.transform.scale(self.map_back, (self.WIDTH, self.HEIGHT))


    # def _render_sidebar(self):
