import gym
from copy import deepcopy
import random
import numpy as np
import math

# project is to use the NEAT algorithm to learn 

# gym is a library to generate the enviornment that the algorithm will run in
env = gym.make('CartPole-v1') # cartpole is the enviornment that this will be tested
observation_space = env.observation_space.shape[0] # integer number for how many inputs are given
action_space = env.action_space.n # how many possible outputs there are in the enviornment 
all_connections = [] # where all connections are stored for innovation numnber

class Node_Genes(): # this is the neuron of the neural network
    def __init__(self):
        self.number = None
        self.layer_number = None
        self.bias = random.uniform(-2, 2)
        self.value = 0
        self.connections_out = []

class Connect_Genes(): # this is what the connection between neurons will look like/where the info will be stored
    def __init__(self):
        self.output_node_number = None
        self.output_node_layer = None
        self.weight = random.uniform(-2, 2)
        self.enabled = True
        self.innovation = None

class Agent(): 
    def __init__(self):
        self.network = [[], []] # network is a 2d array
        self.connections = [] 
        self.connections_weight = []
        self.node_count = 0
        self.all_nodes = []
        self.fitness = 0

        for i in range(observation_space): # making a node for every input the network will take in and adding it to the first level of the array
            node = Node_Genes()
            node.number = i
            node.layer_number = 0
            self.network[0].append(deepcopy(node))
            self.node_count += 1
        for i in range(action_space): # making a node for every output the environment has
            node = Node_Genes()
            node.number = observation_space + i
            node.layer_number = 1
            self.network[1].append(deepcopy(node))
            self.node_count += 1

        # the next part of code is connecting every input node to every output node while also initializing important things about it
        for i in range(len(self.network[0])):
            for j in range(len(self.network[1])):
                connection = Connect_Genes()
                connection.output_node_number = observation_space + j
                connection.output_node_layer = 1
                connection.innovation = self.finding_innovation(self.network[0][i].number, observation_space + j)
                self.connections.append([self.network[0][i].number, observation_space + j])
                self.connections_weight.append(connection.weight)
                self.network[0][i].connections_out.append(deepcopy(connection))
 
    def ReLU(self, x): # this is the relu function which is important in all neural networks
        return x * (x > 0)

    def feed_forward(self, state): # when given a network we want to figure out the outputs to it
        # sweep the entire network setting all values to 0 to make sure nothing is left over
        for i in range(len(self.network)):
            for j in range(len(self.network[i])):
                self.network[i][j].value = 0

        # inputs
        for i in range(len(self.network[0])):
            self.network[0][i].value = state[0][i]

        # entire layer ignoring output
        for i in range(len(self.network)-1):
            # node in each layer
            for j in range(len(self.network[i])):
                # follow connections

                for k in range(len(self.network[i][j].connections_out)):
                    # make sure connection is enabled
                    if self.network[i][j].connections_out[k].enabled == True:
                        output_node_layer = self.network[i][j].connections_out[k].output_node_layer
                        output_node_number = self.network[i][j].connections_out[k].output_node_number
                        weight = self.network[i][j].connections_out[k].weight

                        # find target node    
                        for p in range(len(self.network[output_node_layer])):
                            if self.network[output_node_layer][p].number == output_node_number:
                                self.network[output_node_layer][p].value += self.network[i][j].value * weight # perform the calculation based on the value from the last node connected to the current one

        # each layer thats not first since this is the input layer
        for i in range(1, len(self.network)):
            for j in range(len(self.network[i])):
                self.network[i][j].value = self.ReLU(self.network[i][j].value) + self.network[i][j].bias # normalizing all the values
        
        last_values = []
        maximum_index = len(self.network)-1
        for i in range(len(self.network[maximum_index])):
            last_values.append(self.network[maximum_index][i].value) # returning the last two values of the network
        
        return last_values

    def finding_innovation(self, starting_node_number, ending_node_number): # each connection has an innovation number and this is this part
        # innovation numbers are unique to the conceptual pairing of two nodes and they are unique, innovation number scannot appear twice
        # so we are looking for the connection with that number
        connection_innovation = [starting_node_number, ending_node_number]

        if connection_innovation not in all_connections:
            all_connections.append(connection_innovation)
        innovation_number = all_connections.index(connection_innovation)

        return innovation_number

    def mutate_node(self): # adding node between two connected
        # first block of code picks a random node thats not on the very last layer since you cant connect the last layer to anything else
        random_layer = random.randint(0, len(self.network)-2)
        random_node_index = random.randint(0, len(self.network[random_layer])-1)
        random_node = self.network[random_layer][random_node_index]
        random_connection_index = random.randint(0, len(random_node.connections_out)-1)
        random_connection = random_node.connections_out[random_connection_index]
        
        random_node.connections_out[random_connection_index].enabled = False
        
        output_node_number = random_connection.output_node_number
        output_node_layer = random_connection.output_node_layer
        
        if abs(random_layer - output_node_layer) == 1: # checking that there is a layer that exists already between the two, a new layer has to be added here
            # print("added layer")
            # code here makes a new node and then two connections, one into the node, and one out of the node
            self.network.insert(random_layer + 1, [])
            node = Node_Genes()
            node.number = self.node_count
            self.node_count += 1
            node.layer_number = random_layer + 1
            self.network[random_layer + 1].append(deepcopy(node))
            new_node_index = len(self.network[random_layer + 1]) - 1
        
            connection = Connect_Genes()
            connection.output_node_number = node.number
            connection.output_node_layer = node.layer_number - 1
            connection.innovation = self.finding_innovation(random_node.number, node.number)
            self.connections.append([random_node.number, node.number])
            self.connections_weight.append(connection.weight)
            self.network[random_layer][random_node_index].connections_out.append(deepcopy(connection))

            connection = Connect_Genes()
            connection.output_node_number = output_node_number
            connection.output_node_layer = output_node_layer
            connection.innovation = self.finding_innovation(node.number, output_node_number)
            self.connections.append([node.number, output_node_number])
            self.connections_weight.append(connection.weight)
            self.network[node.layer_number][new_node_index].connections_out.append(deepcopy(connection))

            for i in range(node.layer_number+1, len(self.network)):
                for j in range(len(self.network[i])):
                    self.network[i][j].layer_number += 1
        
            for i in range(len(self.network)):
                for j in range(len(self.network[i])):
                    for k in range(len(self.network[i][j].connections_out)):
                            self.network[i][j].connections_out[k].output_node_layer += 1
            
        else:
            # a new layer does not to be added
            # same method - node made, two connections
            node = Node_Genes()
            node.number = self.node_count
            self.node_count += 1
            node.layer_number = random_layer + 1
            self.network[random_layer + 1].append(deepcopy(node))
            new_node_index = len(self.network[random_layer + 1]) - 1
            
            connection = Connect_Genes()
            connection.output_node_number = node.number
            connection.output_node_layer = node.layer_number
            connection.innovation = self.finding_innovation(random_node.number, node.number)
            self.connections.append([random_node.number, node.number])
            self.connections_weight.append(connection.weight)
            self.network[random_layer][random_node_index].connections_out.append(deepcopy(connection))

            connection = Connect_Genes()
            connection.output_node_number = output_node_number
            connection.output_node_layer = output_node_layer
            connection.innovation = self.finding_innovation(node.number, output_node_number)
            self.connections.append([node.number, output_node_number])
            self.connections_weight.append(connection.weight)
            self.network[node.layer_number][new_node_index].connections_out.append(deepcopy(connection))
            
    def mutate_link(self): # connect two unconnected nodes
        if len(self.network) > 2:
            connected = True

            while connected == True:
                # find two nodes randomly that are not on the same layer and connect them
                first_random_layer = random.randint(0, len(self.network)-2)
                first_random_node_index = random.randint(0, len(self.network[first_random_layer])-1)
                first_random_node = self.network[first_random_layer][first_random_node_index]

                second_random_layer = random.randint(0, len(self.network)-2)
                while first_random_layer == second_random_layer:
                    second_random_layer = random.randint(0, len(self.network)-2)
                second_random_node_index = random.randint(0, len(self.network[second_random_layer])-1)
                second_random_node = self.network[second_random_layer][second_random_node_index]

                #print("node {} layer {} number {}".format(first_random_node, first_random_node.layer_number, first_random_node.number))
                #print("node {} layer {} number {}".format(second_random_node, second_random_node.layer_number, second_random_node.number))
                connected = False
                for i in range(len(first_random_node.connections_out)):
                    if first_random_node.connections_out[i].output_node_number == second_random_node.number:
                        connected = True
                for i in range(len(second_random_node.connections_out)):
                    if second_random_node.connections_out[i].output_node_number == first_random_node.number:
                        connected = True
                
                if connected == False:
                    connection = Connect_Genes()
                    connection.output_node_number = second_random_node.number
                    connection.output_node_layer = second_random_node.layer_number
                    connection.innovation = self.finding_innovation(first_random_node.number, second_random_node.number)
                    self.connections.append([first_random_node.number, second_random_node.number])
                    self.connections_weight.append(connection.weight)
                    self.network[first_random_layer][first_random_node_index].connections_out.append(deepcopy(connection))
                             

    def mutate_enable_disable(self): # randomly pick a node whos connection to enable or disable 
        random_layer = random.randint(0, len(self.network)-2)
        random_node_index = random.randint(0, len(self.network[random_layer])-1)
        random_node = self.network[random_layer][random_node_index]
        random_connection_index = random.randint(0, len(random_node.connections_out)-1)
        
        if random_node.connections_out[random_connection_index].enabled == False:
            self.network[random_layer][random_node_index].connections_out[random_connection_index].enabled = True
        else:
            self.network[random_layer][random_node_index].connections_out[random_connection_index].enabled = False

    def mutate_weight_shift(self):
        #print("mutate weight shift")
        # pick a random node and multiply the weight by a random float between 0 and 2
        random_layer = random.randint(0, len(self.network)-2)
        random_node_index = random.randint(0, len(self.network[random_layer])-1)
        random_node = self.network[random_layer][random_node_index]
        random_connection_index = random.randint(0, len(random_node.connections_out)-1)
        number = random.uniform(0, 2)

        self.network[random_layer][random_node_index].connections_out[random_connection_index].weight *= number
        
        combination = [random_node.number, self.network[random_layer][random_node_index].connections_out[random_connection_index].output_node_number]
        number_index = self.connections.index(combination)
        self.connections_weight[number_index] *= number

    def mutate_weight_random(self):
        # same thing as above except no multiplication, its just being overriden
        random_layer = random.randint(0, len(self.network)-2)
        random_node_index = random.randint(0, len(self.network[random_layer])-1)
        random_node = self.network[random_layer][random_node_index]
        random_connection_index = random.randint(0, len(random_node.connections_out)-1)
        number = random.uniform(-2, 2)

        self.network[random_layer][random_node_index].connections_out[random_connection_index].weight = random.uniform(-2, 2)

        combination = [random_node.number, self.network[random_layer][random_node_index].connections_out[random_connection_index].output_node_number]
        number_index = self.connections.index(combination)
        self.connections_weight[number_index] *= number

    def mutation(self):
        choice = random.randint(0, 10)
        # this is where the mutation happens but also i can make some stuff be more likely to be selected than others
        if choice == 0:
            #print("mutate link")
            self.mutate_link()
        if choice == 1:
            #print("mutate node")
            self.mutate_node()
        if choice == 2 or choice == 5 or choice == 8:
            #print("enable disable")
            self.mutate_enable_disable()
        if choice == 3 or choice == 6 or choice == 9:
            #print("weight shift")
            self.mutate_weight_shift()
        if choice == 4 or choice == 7 or choice == 10:
            #print("weight random")
            self.mutate_weight_random()

    def choose_action(self, state):
        # its in the name, just choose the last value with the biggest value as the action
        last_values = self.feed_forward(state)
        action = np.argmax(last_values)
        return action
    
    def printing_stats(self):
        # this was just for me to print out the entire network and all the connections to see that everything works
        for i in range(len(self.network)):
            print("--")
            print("layer {}".format(i))
            for j in range(len(self.network[i])):
                print("node {} layer {} number {}".format(self.network[i][j], self.network[i][j].layer_number, self.network[i][j].number))

                for k in range(len(self.network[i][j].connections_out)):
                    if self.network[i][j].connections_out[k].enabled == True:
                        print("layer {} number {} innovation {}".format(self.network[i][j].connections_out[k].output_node_layer, self.network[i][j].connections_out[k].output_node_number, self.network[i][j].connections_out[k].innovation))

    def speciation(self, species1, species2):
        # so theres a calculation for how similar two agents are to see if they should be in the same species
        N = max(len(species1.connections), len(species2.connections)) # who has the most connections
        E = abs(species1.node_count - species2.node_count) # difference in node count

        D = 0 # how many connections does each not network share
        both = species1.connections + species2.connections
        for i in range(len(both)):
            if both.count(both[i]) == 1:
                D += 1
        
        W = 0 # sum of weight differences for connections shared
        shorter_species = species1.connections
        if species1.connections <= species2.connections:
            shorter_species = species1.connections

        for i in range(len(shorter_species)):
            connection_identified = shorter_species[i]
            if connection_identified in species1.connections and connection_identified in species2.connections:
                index_species_one = species1.connections.index(connection_identified)
                index_species_two = species2.connections.index(connection_identified)
                
                W += abs(species1.connections_weight[index_species_one] - species2.connections_weight[index_species_two])

        number = E/N + D/N + 0.5*W
        #print(number)
        return number
    
    def making_children(self, species1, species2):
        #print("making children ")
        #print(species1.fitness)
        #print(species2.fitness)

        parents = [species1, species2]
        index = np.argmax([int(species1.fitness), int(species2.fitness)])
        fit_parent = parents[index] # take the fitest parent
        less_fit_parent = parents[abs(1-index)]

        child = Agent() 
        child.network = fit_parent.network # take all the nodes from fittest parent
        child.node_count = fit_parent.node_count 
        
        both = species1.connections + species2.connections
        connections_both = []
        parent_chosen = []
        copy_indexs = []
        both_weights = species1.connections_weight + species2.connections_weight
        
        """
        print("up top")
        print(species1.connections_weight)
        print("some more")
        print(species2.connections_weight)
        """

        for i in range(len(both)): 
            if both.count(both[i]) == 2:
                if both[i] not in connections_both:
                    connections_both.append(both[i])

        for i in range(len(connections_both)): # for connections shared by both its 50/50 of whos you take basically
            # the rest of the code addresses this finding out if its enabled and then taking it 50/50 for connections shared
            parent_chosen.append(random.randint(0, 1))

        #print("connections in both")
        #print(both)
        #print(" ")
        #print(connections_both)
        #print(" ")

        for i in range(len(connections_both)):
            indices = [index for index, element in enumerate(both) if element == connections_both[i]]
            copy_indexs.append(indices)

        #print(copy_indexs)
        #print(parent_chosen)

        for i in range(len(child.network)):
            #print("-- layer {} --".format(i))
            for j in range(len(child.network[i])):
                #print("node {} layer {} number {}".format(child.network[i][j], child.network[i][j].layer_number, child.network[i][j].number))

                for k in range(len(child.network[i][j].connections_out)):
                    if child.network[i][j].connections_out[k].enabled == True:
                        for p in range(len(connections_both)):
                            if connections_both[p][0] == child.network[i][j].number and connections_both[p][1] == child.network[i][j].connections_out[k].output_node_number:
                                """
                                print(connections_both[p])
                                print(copy_indexs[p])
                                print(parent_chosen[p])
                                print(both_weights[copy_indexs[p][parent_chosen[p]]])
                                """
                                child.network[i][j].connections_out[k].weight = both_weights[copy_indexs[p][parent_chosen[p]]]

        child.connections = []
        child.connections_weight = []
        child.all_nodes = []

        for i in range(len(child.network)):
            for j in range(len(child.network[i])):
                child.all_nodes.append(child.network[i][j])

                for k in range(len(child.network[i][j].connections_out)):
                    child.connections.append([child.network[i][j].number, child.network[i][j].connections_out[k].output_node_number])
                    child.connections_weight.append(child.network[i][j].connections_out[k].weight)     

        return child

network_amount = 50
networks = []
threshold_species = 5.5
species = []
average_general_fitness = 0
best_score = float("-inf")
best_agent = None
epochs = 10

for i in range(network_amount):
    agent = Agent()
    for i in range(5): # start by mutating 5 times
        agent.mutation()

    networks.append(deepcopy(agent))

for e in range(epochs): # some amount of generations
    species = []
    average_general_fitness = 0

    for i in range(len(networks)): # play through each network
        agent = networks[i]
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        score = 0

        while True:
            env.render()
            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            state_ = np.reshape(state_, [1, observation_space])
            state = state_
            score += reward

            if done:
                #agent.printing_stats()
                print("score {}".format(score))
                agent.fitness = score # this is important later for making kids
                average_general_fitness += score

                if score > best_score:
                    best_score = score
                    best_agent = agent

                break
    
    average_general_fitness /= network_amount
    networks.append(best_agent) # always keep the best agent
    print("epoch {} best score {} average fitness {}".format(e, best_score, average_general_fitness))
    
    species.append([networks[0]])

    for i in range(1, len(networks)):
        added = False
        for j in range(len(species)):
            if agent.speciation(species[j][0], networks[i]) >= threshold_species: # create the species based on if its higher than the treshold
                species[j].append(networks[i])
                added = True
                break

        if added == False:
            species.append([networks[i]])
    
    # the code here is all about cutting the worse 50%
    for i in range(len(species)):
        species[i].sort(key=lambda x: x.fitness, reverse=True)

    for i in range(len(species)):
        cutting = len(species[i])//2
        new_species = species[i][0:len(species[i]) - cutting]
        species[i] = new_species

    print(len(species))
    # how many kids
    species_average_fitness = []
    new_networks = []

    for i in range(len(species)):
        isolated_average = 0
        for j in range(len(species[i])):
            isolated_average += species[i][j].fitness
        isolated_average /= len(species[i])
        species_average_fitness.append(isolated_average)

        amount = math.ceil(isolated_average/average_general_fitness * len(species[i])) # there is a calculation how many kids each species should have based on how it performs relatively 

        for j in range(amount):
            if amount == 1 or len(species[i]) == 1: # if only one network in species keep it
                new_networks.append(species[i][0])
                break
            else:
                generated = random.sample(species[i], 2) # else randomly make a new kid based off two parents that exist in the species
                first_parent = generated[0]
                second_parent = generated[1]

                child = agent.making_children(first_parent, second_parent) # make child and add it
                new_networks.append(deepcopy(child))

    for i in range(len(new_networks)):
        new_networks[i].mutation() # mutate everyone

    new_networks.append(deepcopy(best_agent))
    networks = new_networks # put the new networks in the networks list and run it again
