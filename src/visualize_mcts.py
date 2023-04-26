import pygame

width = 700
height = 700
window = pygame.display.set_mode((width, height))

pygame.display.set_caption("MTCS Visualization")

BLACK = (0, 0, 0)
GREY = (150, 150, 150)

class Node():
    def __init__(self, x, y, radius, color, parent=None) -> None:
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.parent = parent
        
    def get_color(self) -> tuple:
        return self.color
    
    def get_parent(self) -> "Node":
        return self.parent
    
    def set_parent(self, parent: "Node"):
        self.parent = parent

    def set_color(self, color: tuple):
        self.color = color

    def set_x(self, x: int):
        self.x = x
    
    def set_y(self, y: int):
        self.y = y

def draw_circles(window, width, nodes):
    


    
