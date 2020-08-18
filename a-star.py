import pygame
import math
from random import randint, choice
import numpy as np
import sys

FREE_COLOR = (52, 73, 94)
SEARCHED_COLOR = (32, 53, 74)
OBSTACLE_COLOR = (230, 126, 34)
GOAL_COLOR = (180, 200, 40)
START_COLOR = (39, 174, 96)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

## Holds a map of the world and its current state
#
# Uses a numpy matrix as the map of the world, where given a position x, y its value
# is 0 for an open position, 1 for an obstacle position, 2 for the start position and 3
# for the end/goal position. Its value can also be 4 which is used when drawing the algorithm
# state as an already searched position.
# 
# @params size The square size of the map in positions.
#
# @params obstaclesPercentage A float between 0 and 1. 
class World():
    def __init__(self, size, obstaclesPercentage):
        self.size = (size, size)

        self.buildWorld()

        n_obstacles = int(obstaclesPercentage*size*size)
        self.generateObstacles(n_obstacles)

    ## Changes start position of the search.
    #
    # @params A tuple of the format (x, y).
    def setStartPos(self, pos):
        self.map[self.startpos[1], self.startpos[0]] = 0 # "erase" last start pos
        self.startpos = pos     # update current startpos
        self.map[pos[1], pos[0]] = 2 # writes startpos to map

    ## Changes end/goal position of the search.
    #
    # @params A tuple of the format (x, y).
    def setEndPos(self, pos):
        self.map[self.endpos[1], self.endpos[0]] = 0
        self.endpos = pos
        self.map[pos[1], pos[0]] = 3

    ## Construct a square map of the given size.
    #
    # Builds a "world" which is stored as a Numpy array. It uses
    # the size given in the constructor a returns a matrix size x size
    # initializing every position with 0.
    def buildWorld(self):
        # holds the world map, 0 means empty space and 1 obstacles
        self.map = np.zeros(self.size) 
        self.startpos = (0, 0)
        self.map[0, 0] = 2

        self.endpos = (self.size[1]-1, self.size[0]-1)
        self.map[self.size[1]-1, self.size[0]-1] = 3

    ## Generate n percentage of obstacles randomingly.
    #
    # Randomingly generate obstacles by filling the map with 1's where there are
    # open positions, until the given percentage of obstacles is met.
    #
    # @params n A float between 0 and 1. The percentage of map to be filled
    # with obstacles.
    def generateObstacles(self, n):
        free_spaces = [] # keeps track of free spaces
        for y in range(0, self.size[1]):
            for x in range(0, self.size[0]):
                if (x, y) != self.startpos and (x, y) != self.endpos:
                    free_spaces.append((x, y))

        for i in range(0, n):
            # gets a space from free_spaces
            space = choice(free_spaces)
            self.map[space[1], space[0]] = 1 # because numpython
            free_spaces.remove(space)

    ## Reset world without removing obstacles.
    #
    # While the A* search is running positions that have been searched already
    # are filled with 4's. This method effectively "cleans" the map by replacing
    # the 4's with 0's without removing obstacles.
    def resetWorld(self):
        for y in range(0, self.size[1]):
            for x in range(0, self.size[0]):
                if self.map[y, x] == 4:
                    self.map[y, x] = 0

## Performs A* search step-by-step.
#
# Implements A* search algorithm with methods to perform it in a step-by-step
# approach.
#
# @params world A World object, which contains a map and methods for modifying it.
class AStarStep():
    
    def __init__(self, world):
        self.world = world

    ## Initializes variables, lists and dictionaries.
    #
    # Initializes all things needed by the algorithm to perform the search.
    # This is needed in a separate function because I am not running the search
    # in a loop.
    def initStepSearch(self):
        self.start = self.world.startpos
        self.goal = self.world.endpos
        print("Start:", self.start, "Goal:", self.goal)
        # lists all possible spaces, used to fill the dictionaries needed
        # for the search algorithm
        spaces = [(x, y) for x in range(0, self.world.size[1]) for y in range(0, self.world.size[0])]

        self.closedSet = [] # nodes that have already been visited

        self.openSet = [self.start] # starts with only the first node
        # for each node, which node it can most efficiently
        # be reached from
        self.cameFrom = {}
        #for space in spaces:
        #    cameFrom[space] = (1, 1)

        # for each node, the cost of getting from the start node to that node.
        self.gScore = {}
        for space in spaces:
            self.gScore[space] = float('inf')
        self.gScore[self.start] = 0.0 # cost of going from start to start is 0

        # for each node, the total cost of getting from the start node to the
        # goal by passing that node.
        self.fScore = {}
        for space in spaces:
            self.fScore[space] = float('inf')
        self.fScore[self.start] = self.calcHeuristic(self.start, self.goal)

        self.done = False    # keeps track if algorithm has finished
        self.success = False # is True if algorithm has finished succesfully

    ## Performs the A star search step-by-step each time the method is called.
    #
    # To make it easier to show the progression of the algorithm in PyGame the
    # search is implemented so it progresses a "step" each time the method is called.
    def stepSearch(self):
        if self.done:
            print('Finished:', self.success)

        elif self.openSet: # while openSet is not empty
            self.current = self.cheapestNode(self.fScore)

            if self.current == self.goal: # algorithm has finished
                self.done = True
                self.success = True

            else:    # still hasnt find best path
                self.openSet.remove(self.current) # remove current from openSet
                self.closedSet.append(self.current) # add current to closedSet

                # get neighbors of current node
                neighbors = self.getNeighbors(self.current)
                for n in neighbors:
                    if n in self.closedSet:
                        continue # neighbors has already been checked

                    # distance from start to neighbor
                    n_gScore = self.gScore[self.current] + 1 # fixed cost of 1 per step
                    #print current, n, n_gScore
                    if n not in self.openSet:
                        self.openSet.append(n) # adds this neighbor to openSet
                    elif n_gScore >= self.gScore[n]:
                        continue # this not the better path

                    self.cameFrom[n] = self.current
                    self.gScore[n] = n_gScore
                    self.fScore[n] = self.gScore[n] + self.calcHeuristic(n, self.goal)
        else:
            self.done = True
            self.success = False

    ## Reconstruct path from given node.
    #
    # Returns a list of the nodes in the path from the given node and the start
    # node.
    #
    # @params node Node from which to traceback to start node.
    def reconstructPath(self, node):
        total_path = [node]
        while node in self.cameFrom.keys():
            node = self.cameFrom[node]
            total_path.append(node)
        return total_path

    ## Calculates a heuristic approximation of getting from nodes.
    #
    # Calculates the distance in a straight line from node1 to node2 (usually goal node).
    # @params node1 First node coordinate.
    # @params node2 Second node coordinate.
    def calcHeuristic(self, node1, node2):
        x1, y1 = node1
        x2, y2 = node2

        # distance in a straight line
        linearDist = math.sqrt(((x2-x1) ** 2 + (y2-y1) ** 2))

        #print node1, 'to goal:', linearDist

        return linearDist

    ## Returns all valid neighbors of the given node.
    #
    # Checks the position around the given node, excluding diagonal positions,
    # and returns the valid ones, i.e. position which don't contain 1.
    #
    # @params The node from which to return the neighbors.
    def getNeighbors(self, node):
        x, y = node
        neighbors = []

        # determining neighbors
        if x+1 < self.world.size[1] and self.world.map[y, x + 1] != 1:
            neighbors.append((x + 1, y))
        if x > 0 and self.world.map[y, x - 1] != 1:
            neighbors.append((x - 1, y))
        if y+1 < self.world.size[0] and self.world.map[y + 1, x] != 1:
            neighbors.append((x, y+1))
        if y > 0 and self.world.map[y - 1, x] != 1:
            neighbors.append((x, y - 1))

        return neighbors

    ## Return the cheapest node in the openSet given the given scores.
    #
    ## @params scores Dictionary of the scores of each node.
    def cheapestNode(self, scores):
        chinest_node = self.openSet[0]
        cheapest_score = scores[chinest_node]
        for s in self.openSet[1:]:
            if scores[s] < cheapest_score:
                cheapest_score = scores[s]
                chinest_node = s

        return chinest_node

## A visualizer for the A* algorithm.
#
# Implements a graphical interface showing the progress of the algorithm while
# trying to find the optimal path in the world. It also provides text help on
# the screen and handles keyboard inputs to give commands like start, perform
# a single step of the search and reset the world.
#
# @params searchTree An AStartStep object which will be called to execute the
# algorithm and draw its state on the screen.
#
# @params squarePixelSize The size of the drawn square/position in pixels.
class WorldRenderer():
    
    def __init__(self, searchTree, squarePixelSize):
        self.searchTree = searchTree

        self.runSearch = False
        self.searchInit = False

        self.posSize = squarePixelSize # size of each position

        self.path = None # when and if a solution is found this where the path
                         # to the goal is stored
        pygame.init() # initialize pygame

        self.textSurface = self.createTextSurface()

        self.screenSize = tuple([self.posSize*i for i in self.searchTree.world.size]) # gets screen size
        self.screen = pygame.display.set_mode(self.screenSize) # the thing we draw things on

    ## Creates a text surface.
    #
    # Crates the splash text first shown when running the program, showing info
    # about commands.
    def createTextSurface(self):
        font = pygame.font.SysFont('Oxygen Mono', 30)
        outline = pygame.font.SysFont('Oxygen Mono', 30)

        infoText = "S - Start/Pause search\n"
        infoText += "D - Execute one step\n"
        infoText += "R - Reset map\n"
        infoText += "Left Click - Set start position\n"
        infoText += "Right Click - Set goal position"

        lines = infoText.splitlines()
        w = h = 0

        for l in lines:
            w = max(w, font.size(l)[0])
            h += font.get_linesize()
        
        surf = pygame.Surface((w, h), pygame.SRCALPHA, 32)
        h = 0
        for l in lines:
            t = font.render(l, True, WHITE)
            o = outline.render(l, True, BLACK)
            surf.blit(o, (0, h))
            surf.blit(t, (0, h))
            h += font.get_linesize()

        #self.textSurface.blit(font.render(infoText, True, WHITE), (0, 30))
        #self.textSurface = font.render(infoText, True, WHITE)
        return surf

    ## Handles keyboard input
    #
    # Uses PyGame events to handle keyboard and mouse input for starting or pausing the
    # search, resetting the search, performing a single step of the search, and
    # changing the starting and ending/goal positions.
    def handleInput(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self.runSearch = not self.runSearch # changes running state
                elif event.key == pygame.K_r:
                    self.resetSearch() # resets search
                elif event.key == pygame.K_d:
                    if self.searchInit == False:
                        self.searchTree.initStepSearch()
                        self.searchInit = True
                    self.searchTree.stepSearch() # performs a single step
                    self.path = self.searchTree.reconstructPath(self.searchTree.current)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_buttons = pygame.mouse.get_pressed()
                if mouse_buttons[0] == 1:
                    new_pos = pygame.mouse.get_pos()
                    new_pos = tuple([i // self.posSize for i in new_pos])
                    self.searchTree.world.setStartPos(new_pos)
                    self.resetSearch()

                elif mouse_buttons[2] == 1:
                    new_pos = pygame.mouse.get_pos()
                    new_pos = tuple([i // self.posSize for i in new_pos])
                    self.searchTree.world.setEndPos(new_pos)
                    self.resetSearch()

    ## Resets the searchTree object.
    #
    # Restarts the map and search to be ready to perform a new search. Called
    # when a key is pressed or start or end positions are changed.
    def resetSearch(self):
        self.runSearch = False # stop search if it is running
        self.searchTree.initStepSearch()
        self.searchTree.world.resetWorld() # cleans positions marked as searched
        self.path = None # clears path

    ## Draws the last searched path.
    #
    # Performs a "pretty" animation showing the last path which was searched,
    # providing a nice visualization for how the algorithm works and progresses
    # through the world.
    def drawPath(self):
        p0 = self.path[0]
        startpos = (p0[0] * self.posSize + self.posSize/2, p0[1] * self.posSize + self.posSize/2)
        for p in self.path:
            endpos = (p[0] * self.posSize+self.posSize/2, p[1] * self.posSize + self.posSize/2)
            pygame.draw.line(self.screen, START_COLOR, startpos, endpos, 3)
            startpos = endpos

            if p != self.searchTree.world.startpos and p != self.searchTree.world.endpos:
                self.searchTree.world.map[p[1], p[0]] = 4

    ## Draws the current state of the world.
    #
    # Iterates through every position in the searchTree world and draws
    # a rectangle according to its value.
    # 0 - Open position.
    # 1 - Obstacle.
    # 2 - Start position for the search.
    # 3 - Goal position.
    # 4 - Searched position.
    def drawWorld(self):
        for y in range(0, self.searchTree.world.size[0]):
            for x in range(0, self.searchTree.world.size[1]):
                rect = pygame.Rect(self.posSize*x, self.posSize*y,
                                    self.posSize, self.posSize)
                #if self.world.map[y, x] == 0:
                #    pygame.draw.rect(self.screen, FREE_COLOR, rect)
                if self.searchTree.world.map[y, x] == 1: # Obstacle
                    pygame.draw.rect(self.screen, OBSTACLE_COLOR, rect)
                elif self.searchTree.world.map[y, x] == 2: # Start position
                    pygame.draw.rect(self.screen, START_COLOR, rect)
                elif self.searchTree.world.map[y, x] == 3: # End position / goal
                    pygame.draw.rect(self.screen, GOAL_COLOR, rect)
                elif self.searchTree.world.map[y, x] == 4: # Searched position
                    pygame.draw.rect(self.screen, SEARCHED_COLOR, rect)

    ## Draws the info text.
    #
    # Draws text info about mouse and keyboard inputs.
    def drawInfoText(self):
        self.screen.blit(self.textSurface, (50, 250))

    ## Main loop of renderer.
    #
    # Takes care of calling everything necessary to run, control and draw
    # the algorithm on the screen.
    def run(self):
        while True:
            self.drawScreen()
            self.handleInput()

            if self.runSearch == True: # search is running
                if self.searchInit == False: # if initStepSearch hasnt been called yet
                    self.searchTree.initStepSearch()  # calls it
                    self.searchInit = True
                elif self.searchTree.done: # search has finished
                    self.runSearch = False
                    self.searchInit = False
                else: # search hasnt finished yet
                    self.searchTree.stepSearch() # simply calls stepSearch to perform an iteration
                    self.path = self.searchTree.reconstructPath(self.searchTree.current)

    ## Refreshes and draws the screen.
    #
    # Calls every function necessary to show the current state of the search
    # on the screen.
    def drawScreen(self):
        self.screen.fill(FREE_COLOR)

        self.drawWorld()

        if self.path:
            self.drawPath()
        else:
            self.drawInfoText()

        pygame.display.flip()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage:", sys.argv[0], "<square-map-size> <obstacle-percentage>")
        sys.exit(1)

    map_size = int(sys.argv[1])
    obstacle_percentage = float(sys.argv[2])
    screen_size = (1366, 768)
    square_pixel_size = (min(screen_size)-100) // map_size

    world = World(map_size, obstacle_percentage)
    #search = AStarSearch(world)
    search = AStarStep(world)
    renderer = WorldRenderer(search, square_pixel_size)
    renderer.run()
