import pygame

window_size = (700, 700)
window = pygame.display.set_mode(window_size)

pygame.display.set_caption("MTCS Visualization")

BLACK = (0, 0, 0)
GREY = (150, 150, 150)

class Node():
    def __init__(self, pos=None, parent=None) -> None:
        self.radius = 40
        self.color = BLACK
        self.pos = pos
        self.parent = parent
        self.children = []

        if (parent != None):
            self.parent.add_child(self)
        
    def get_color(self) -> tuple:
        return self.color
    
    def get_parent(self) -> "Node":
        return self.parent
    
    def get_children(self) -> list:
        return self.children

    def get_radius(self) -> int:
        return self.radius
    
    def get_pos(self) -> tuple:
        return self.pos
    
    def set_parent(self, parent: "Node"):
        self.parent = parent
        self.parent.add_child(self)

    def set_color(self, color: tuple):
        self.color = color

    def set_pos(self, pos: tuple):
        self.pos = pos
    
    def set_radius(self, radius: int):
        self.radius = radius

    def add_child(self, child):
        self.children.append(child)

def draw_nodes(root_node: "Node") -> None:
    '''
    Draws all the nodes given by root node and children.
    '''
    root_children = root_node.get_children()
    root_color = root_node.get_color()
    root_position = root_node.get_pos()
    root_radius = root_node.get_radius()
    root_node.set_radius(root_radius)
    root_node.set_pos(root_position)

    pygame.draw.circle(window, root_color, root_position, root_radius)
    
    if (len(root_children) == 0):
        return
    
    space = 0
    for child in root_children:
        '''
        print(root_node.get_pos()[0])
        print(root_node.get_pos()[0] - child.get_radius()*(len(root_children) - 1) + child.get_radius()*space)
        print(root_node.get_pos()[1])
        print(root_node.get_pos()[1] + child.get_radius())
        '''
        child.set_radius(root_radius//2)
        child.set_pos(((root_node.get_pos()[0] - 2*child.get_radius()*(len(root_children) - 1)) + 4*child.get_radius()*space, root_node.get_pos()[1] + 2*root_node.get_radius()))
        draw_nodes(child)
        space += 1

def draw_lines(root_node: Node) -> None:
    '''
    Renders the lines between all nodes in the tree.
    '''
    root_children = root_node.get_children()

    if (len(root_children) == 0):
        return

    for child_node in root_children:
        pygame.draw.line(window, BLACK, root_node.get_pos(), child_node.get_pos(), 2)
        draw_lines(child_node)

    
def draw_tree(root_node: "Node") -> None:
    '''
    Draws the tree itself.
    '''
    draw_nodes(root_node)
    draw_lines(root_node)
    pygame.display.update()
    
def main(root: Node) -> None:
    pygame.init()
    window.fill(GREY)
    pygame.display.update()
    draw_tree(root)

    while True:
        pass

if __name__ == "__main__":
    root = Node((window_size[0]//2 , window_size[1]//2))
    root_child_0_0 = Node(parent=root)
    root_child_0_1 = Node(parent=root)
    root_child_0_2 = Node(parent=root)
    root_child_1_0 = Node(parent=root_child_0_2)
    root_child_1_1 = Node(parent=root_child_0_2)
    main(root)



    
