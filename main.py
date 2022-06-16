from cv2 import threshold
import gym
from copy import deepcopy
import random
import numpy as np
import math

env = gym.make('CartPole-v1')
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
all_connections = []

class Node_Genes():
    def __init__(self):
        self.number = None
        self.layer_number = None
        self.bias = random.uniform(-2, 2)
        self.value = 0
        self.connections_out = []

class Connect_Genes():
    def __init__(self):
        self.output_node_number = None
        self.output_node_layer = None
        self.weight = random.uniform(-2, 2)
        self.enabled = True
        self.innovation = None

class Agent():
    def __init__(self):
        self.network = [[], []]
        self.connections = []
        self.connections_weight = []
        self.node_count = 0
        self.all_nodes = []
        self.fitness = 0

        for i in range(observation_space):
            node = Node_Genes()
            node.number = i
            node.layer_number = 0
            self.network[0].append(deepcopy(node))
            self.node_count += 1
        for i in range(action_space):
            node = Node_Genes()
            node.number = observation_space + i
            node.layer_number = 1
            self.network[1].append(deepcopy(node))
            self.node_count += 1

        for i in range(len(self.network[0])):
            for j in range(len(self.network[1])):
                connection = Connect_Genes()
                connection.output_node_number = observation_space + j
                connection.output_node_layer = 1
                connection.innovation = self.finding_innovation(self.network[0][i].number, observation_space + j)
                self.connections.append([self.network[0][i].number, observation_space + j])
                self.connections_weight.append(connection.weight)
                self.network[0][i].connections_out.append(deepcopy(connection))

    def ReLU(self, x):
        return x * (x > 0)

    def feed_forward(self, state):
        # sweep the entire network setting all values to 0
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
                                self.network[output_node_layer][p].value += self.network[i][j].value * weight

        # each layer thats not first
        for i in range(1, len(self.network)):
            for j in range(len(self.network[i])):
                self.network[i][j].value = self.ReLU(self.network[i][j].value) + self.network[i][j].bias
        
        last_values = []
        maximum_index = len(self.network)-1
        for i in range(len(self.network[maximum_index])):
            last_values.append(self.network[maximum_index][i].value)
        
        return last_values

    def finding_innovation(self, starting_node_number, ending_node_number):
        connection_innovation = [starting_node_number, ending_node_number]

        if connection_innovation not in all_connections:
            all_connections.append(connection_innovation)
        innovation_number = all_connections.index(connection_innovation)

        return innovation_number

    def mutate_node(self): # adding node between two connected
        random_layer = random.randint(0, len(self.network)-2)
        random_node_index = random.randint(0, len(self.network[random_layer])-1)
        random_node = self.network[random_layer][random_node_index]
        random_connection_index = random.randint(0, len(random_node.connections_out)-1)
        random_connection = random_node.connections_out[random_connection_index]
        
        random_node.connections_out[random_connection_index].enabled = False
        
        output_node_number = random_connection.output_node_number
        output_node_layer = random_connection.output_node_layer
        
        if abs(random_layer - output_node_layer) == 1:
            # print("added layer")
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
            # print("minimum layer")
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
                             

    def mutate_enable_disable(self):
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
        #self.printing_stats()
        #print("--")
        #print(self.connections)
        #print("--")
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
        last_values = self.feed_forward(state)
        action = np.argmax(last_values)
        return action
    
    def printing_stats(self):
        for i in range(len(self.network)):
            print("--")
            print("layer {}".format(i))
            for j in range(len(self.network[i])):
                print("node {} layer {} number {}".format(self.network[i][j], self.network[i][j].layer_number, self.network[i][j].number))

                for k in range(len(self.network[i][j].connections_out)):
                    if self.network[i][j].connections_out[k].enabled == True:
                        print("layer {} number {} innovation {}".format(self.network[i][j].connections_out[k].output_node_layer, self.network[i][j].connections_out[k].output_node_number, self.network[i][j].connections_out[k].innovation))

    def speciation(self, species1, species2):
        N = max(len(species1.connections), len(species2.connections))
        E = abs(species1.node_count - species2.node_count)

        D = 0
        both = species1.connections + species2.connections
        for i in range(len(both)):
            if both.count(both[i]) == 1:
                D += 1
        
        W = 0
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

        """
        self.network = [[], []]
        self.connections = []
        self.connections_weight = []
        self.node_count = 0
        self.all_nodes = []
        self.fitness = 0
        """

        parents = [species1, species2]
        index = np.argmax([int(species1.fitness), int(species2.fitness)])
        fit_parent = parents[index]
        less_fit_parent = parents[abs(1-index)]

        child = Agent()
        child.network = fit_parent.network
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

        for i in range(len(connections_both)):
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
    for i in range(5):
        agent.mutation()

    networks.append(deepcopy(agent))

for e in range(epochs):
    species = []
    average_general_fitness = 0

    for i in range(len(networks)):
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
                agent.fitness = score
                average_general_fitness += score

                if score > best_score:
                    best_score = score
                    best_agent = agent

                break
    
    average_general_fitness /= network_amount
    networks.append(best_agent)
    print("epoch {} best score {} average fitness {}".format(e, best_score, average_general_fitness))
    
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

        amount = math.ceil(isolated_average/average_general_fitness * len(species[i]))

        for j in range(amount):
            if amount == 1 or len(species[i]) == 1:
                new_networks.append(species[i][0])
                break
            else:
                generated = random.sample(species[i], 2)
                first_parent = generated[0]
                second_parent = generated[1]

                child = agent.making_children(first_parent, second_parent)
                new_networks.append(deepcopy(child))

    for i in range(len(new_networks)):
        new_networks[i].mutation()

    new_networks.append(deepcopy(best_agent))
    networks = new_networks