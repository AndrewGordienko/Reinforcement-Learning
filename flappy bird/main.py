import pygame
import sys
import random
import math
from copy import deepcopy
import random
import numpy as np

from neat import Agent

pygame.init()

WIDTH = 600
HEIGHT = 400
SIZE = WIDTH, HEIGHT
BLACK = 0, 0, 0
WHITE = 255, 255, 255
screen = pygame.display.set_mode(SIZE)
GAP = 100

class Pipe:
    def __init__(self):
        self.scroll = 10
        self.pipe_amount = 3
        self.pipes = []
        self.pipe_xs = []
        self.space = 200
        self.pipe_counter = 0
        self.pipe_objects_top = []
        self.pipe_objects_bottom = []
        self.counter_end = 0
        self.new_pipe = []

        for i in range(self.pipe_amount):
            self.pipes.append(self.distances())
            self.pipe_xs.append(self.space + self.space * i)


    def distances(self):
        difference = HEIGHT - GAP
        standard = difference/2

        pipe1, pipe2 = standard, standard

        increment = random.randint(0, 10)
        growing = random.randint(0, 1)

        if growing == 0:
            pipe1 += increment * 10
            pipe2 -= increment * 10
        else:
            pipe2 += increment * 10
            pipe1 -= increment * 10
        
        return [pipe1, pipe2]
    
    def draw(self):
        infront = False
        self.pipe_objects_top = []
        self.pipe_objects_bottom = []

        for i in range(len(self.pipes)):
            pipe1, pipe2 = self.pipes[i][0], self.pipes[i][1]
        
            pygame.draw.rect(screen, WHITE, pygame.Rect(self.pipe_xs[i], 0, 50, pipe1))
            pygame.draw.rect(screen, WHITE, pygame.Rect(self.pipe_xs[i], HEIGHT - pipe2, 50, pipe2))

            self.pipe_objects_top.append((self.pipe_xs[i], 0, 50, pipe1))
            self.pipe_objects_bottom.append((self.pipe_xs[i], HEIGHT - pipe2, 50, pipe2))

            self.pipe_xs[i] -= self.scroll

            if self.pipe_xs[i] + 25 <= bird.bird_x:
                if len(self.pipes) <= max_score:
                    self.pipes.append(self.distances())
                    self.pipe_xs.append(self.space + self.space * (len(self.pipes)-1))

            else:
                if infront == False:
                    infront = True
                    bird.passed = i
        
        
        

class Bird:
    def __init__(self):
        self.bird_x = 275
        self.bird_y = HEIGHT/2
        self.dimension = 20
        self.v_x = 10
        self.v_y = 0
        self.hop = -40
        self.gravity = 10
        self.passed = 0
        self.d1 = None
        self.d2 = None
        self.d3 = None
        self.d4 = None
    
    def draw(self):
        self.bird_y += self.gravity
        self.bird_y += self.v_y
    
        pygame.draw.rect(screen, WHITE, pygame.Rect(self.bird_x, self.bird_y, self.dimension, self.dimension))

        if self.v_y == self.hop:
            self.v_y = 0

        starting_position = (self.bird_x + self.dimension/2, self.bird_y + self.dimension/2)
        ending_position = (pipe.pipe_xs[self.passed], pipe.pipes[self.passed][0])
        pygame.draw.line(screen, (255, 0, 0), starting_position, ending_position)
        self.d1 = math.sqrt((abs(self.bird_x - pipe.pipe_xs[self.passed]))**2 + (abs(self.bird_y - pipe.pipes[self.passed][0]))**2)

        starting_position = (self.bird_x + self.dimension/2, self.bird_y + self.dimension/2)
        ending_position = (pipe.pipe_xs[self.passed], pipe.pipes[self.passed][0] + GAP)
        pygame.draw.line(screen, (255, 0, 0), starting_position, ending_position)
        self.d2 = math.sqrt((abs(self.bird_x - pipe.pipe_xs[self.passed]))**2 + (abs(self.bird_y - (pipe.pipes[self.passed][0] + GAP)))**2)
        
        if self.passed != 0:
            starting_position = (self.bird_x + self.dimension/2, self.bird_y + self.dimension/2)
            ending_position = (pipe.pipe_xs[self.passed - 1] + 50, pipe.pipes[self.passed - 1][0])
            pygame.draw.line(screen, (255, 0, 0), starting_position, ending_position)
            self.d3 = math.sqrt((abs(self.bird_x - pipe.pipe_xs[self.passed - 1]))**2 + (abs(self.bird_y - pipe.pipes[self.passed - 1][0]))**2)

            starting_position = (self.bird_x + self.dimension/2, self.bird_y + self.dimension/2)
            ending_position = (pipe.pipe_xs[self.passed - 1] + 50, pipe.pipes[self.passed - 1][0] + GAP)
            pygame.draw.line(screen, (255, 0, 0), starting_position, ending_position)
            self.d4 = math.sqrt((abs(self.bird_x - pipe.pipe_xs[self.passed - 1] + 50))**2 + (abs(self.bird_y - (pipe.pipes[self.passed - 1][0] + GAP)))**2)
        else:
            self.d3 = self.d1
            self.d4 = self.d2

        distances = (self.d1, self.d2, self.d3, self.d4)
    
    def collision(self):
        collide = False
        bird_unit = pygame.Rect(self.bird_x, self.bird_y, self.dimension, self.dimension)

        pygame.draw.rect(screen, (255, 0, 0), pipe.pipe_objects_top[self.passed])
        object = pygame.Rect(pipe.pipe_objects_top[self.passed])
        if collide == False: collide = object.contains(bird_unit)

        pygame.draw.rect(screen, (255, 0, 0), pipe.pipe_objects_bottom[self.passed])
        object = pygame.Rect(pipe.pipe_objects_bottom[self.passed])
        if collide == False: collide = object.contains(bird_unit)

        if self.passed != 0:
            # pipe behind
            pygame.draw.rect(screen, (0, 255, 0), pipe.pipe_objects_top[self.passed - 1])
            object = pygame.Rect(pipe.pipe_objects_top[self.passed - 1])
            if collide == False: collide = object.contains(bird_unit)

            pygame.draw.rect(screen, (0, 255, 0), pipe.pipe_objects_bottom[self.passed - 1])
            object = pygame.Rect(pipe.pipe_objects_bottom[self.passed - 1])
            if collide == False: collide = object.contains(bird_unit)

        if self.bird_y <= 0 or self.bird_y >= HEIGHT:
            collide = True

        #print("have we collided? {} score is {}".format(collide, self.passed))
        return collide



network_amount = 50
networks = []
epochs = 50
best_score = float("-inf")
best_agent = None
greed_exponent = 2
threshold_species = 4
max_score = 500

for i in range(network_amount):
    agent = Agent()
    for i in range(5):
        agent.mutation()
    networks.append(deepcopy(agent))

best_agents = []
for i in range(int(network_amount * 0.1)):
    agent = Agent()
    agent.fitness = float("-inf")
    best_agents.append(deepcopy(agent))

for e in range(epochs):
    print("generation {}".format(e))
    species = []
    average_general_fitness = 0

    local_best_agents = []
    for i in range(int(network_amount * 0.1)):
        agent = Agent()
        agent.fitness = float("-inf")
        local_best_agents.append(deepcopy(agent))

    for i in range(len(networks)):
        agent = networks[i]
        pipe = Pipe()
        bird = Bird()
        score = 0

        screen.fill(BLACK)
        pipe.draw()
        bird.draw()
        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        bird.v_y = bird.hop
            
            state = [bird.d1, bird.d2, bird.d3, bird.d4, bird.bird_y]
            action = agent.choose_action(state)
            score = bird.passed

            if action == 1:
                bird.v_y = bird.hop

            screen.fill(BLACK)
            pipe.draw()
            bird.draw()

            finished = False
            if score == max_score:
                finished = True

            if bird.collision() == True or finished == True:
                #print("we have collided")
                score = bird.passed
                agent.fitness = bird.passed
                average_general_fitness += bird.passed

                best_agents.sort(key=lambda x: x.fitness)
                if best_agents[0].fitness < agent.fitness:
                    best_fitnesss = []
                    best_agents[0] = deepcopy(agent)
                best_agents.sort(key=lambda x: x.fitness)

                local_best_agents.sort(key=lambda x: x.fitness)
                if local_best_agents[0].fitness < agent.fitness:
                    local_best_agents[0] = deepcopy(agent)
                local_best_agents.sort(key=lambda x: x.fitness)

                if score > best_score:
                    best_score = score
                    best_agent = deepcopy(agent)
                print(score)

                break

            pygame.display.flip()
            pygame.time.delay(100)
    
    average_general_fitness /= len(networks)
    networks.append(deepcopy(best_agent))

    print("epoch {} best score {} average fitness {}".format(e, best_score, average_general_fitness))
    
    local_best_fitness = []
    for i in range(len(local_best_agents)):
        local_best_fitness.append(deepcopy(local_best_agents[i].fitness))
    best_fitnesss = []
    for i in range(len(best_agents)):
        best_fitnesss.append(deepcopy(best_agents[i].fitness))
    print("local best fitnesss {}".format(local_best_fitness))
    print("best overall fitness {}".format(best_fitnesss))


    species.append([networks[0]])

    for i in range(1, len(networks)):
        added = False
        for j in range(len(species)):
            if agent.speciation(species[j][0], networks[i]) >= threshold_species:
                species[j].append(networks[i])
                added = True
                break

        if added == False:
            species.append([networks[i]])

    for i in range(len(species)):
        species[i].sort(key=lambda x: x.fitness, reverse=True)

    for i in range(len(species)):
        cutting = len(species[i])//2
        new_species = species[i][0:len(species[i]) - cutting]
        species[i] = new_species

    print("dividing into {} species".format(len(species)))
    # how many kids
    species_average_fitness = []
    new_networks = []

    for i in range(len(species)):
        isolated_average = 0
        for j in range(len(species[i])):
            isolated_average += species[i][j].fitness
        isolated_average /= len(species[i])
        species_average_fitness.append(isolated_average)

        amount = math.ceil(isolated_average/average_general_fitness * len(species[i]))

        for j in range(len(species[i])):
            species[i][j].fitness *= greed_exponent

        for j in range(amount):
            if amount == 1 or len(species[i]) == 1:
                new_networks.append(species[i][0])
                break
            else:
                generated = []
                generated.append(int(agent.selecting_score(species[i])))

                second_index = agent.selecting_score(species[i])
                while second_index == generated[0]:
                    second_index = agent.selecting_score(species[i])
                
                generated.append(int(second_index))

                first_parent = species[i][generated[0]]
                second_parent = species[i][generated[1]]

                child = agent.making_children(first_parent, second_parent)
                new_networks.append(deepcopy(child))
    
    new_networks += deepcopy(local_best_agents)
    new_networks += deepcopy(best_agents)
    
    for i in range(len(new_networks)):
        new_networks[i].mutation()

    new_networks.append(deepcopy(best_agent))
    new_networks += deepcopy(local_best_agents)
    new_networks += deepcopy(best_agents)
    
    if len(new_networks) < network_amount:
        for i in range(abs(network_amount - len(new_networks))):
            agent = Agent()
            for i in range(5):
                agent.mutation()
            new_networks.append(deepcopy(agent))

    networks = new_networks
    threshold_species += 0.1 * (5 - len(species))
