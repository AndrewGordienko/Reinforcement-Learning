from copy import deepcopy
import random
import numpy as np
import math

observation_space = 3
action_space = 2
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
            self.network[0][i].value = state[i]

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

            if i != 0:
                for j in range(len(self.network[i])):
                    self.network[i][j].value = self.ReLU(self.network[i][j].value) + self.network[i][j].bias
        
        i += 1
        for j in range(len(self.network[i])):
            self.network[i][j].value = self.ReLU(self.network[i][j].value) + self.network[i][j].bias
        
        last_values = []
        maximum_index = len(self.network)-1
        for i in range(len(self.network[maximum_index])):
            last_values.append(np.tanh(self.network[maximum_index][i].value))
        
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
            
    def mutate_link(self): # connect two unconnected nodes possible infinite loop exists here
        if len(self.network) > 2:
            connected = True
            bypass = 0

            while connected == True and bypass == 0:                    
                first_random_layer = random.randint(0, len(self.network)-2)
                first_random_node_index = random.randint(0, len(self.network[first_random_layer])-1)
                first_random_node = self.network[first_random_layer][first_random_node_index]

                counter = 0
                second_random_layer = random.randint(0, len(self.network)-2)
                while first_random_layer == second_random_layer:
                    second_random_layer = random.randint(0, len(self.network)-2)
                    counter += 1

                    if counter == len(self.all_nodes):
                        bypass = 1
                        break
                second_random_node_index = random.randint(0, len(self.network[second_random_layer])-1)
                second_random_node = self.network[second_random_layer][second_random_node_index]

                connected = False
                for i in range(len(first_random_node.connections_out)):
                    if first_random_node.connections_out[i].output_node_number == second_random_node.number:
                        connected = True
                for i in range(len(second_random_node.connections_out)):
                    if second_random_node.connections_out[i].output_node_number == first_random_node.number:
                        connected = True
                
                if connected == False and bypass == 0:
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

    def selecting_score(self, all_parents):
        fitness_scores = []
        for i in range(len(all_parents)):
            fitness_scores.append(all_parents[i].fitness)

        probabilities = []
        added_probs = []
        total_score = 0

        for i in range(len(fitness_scores)):
            total_score += fitness_scores[i]
        factor = 1/total_score

        for i in range(len(fitness_scores)):
            probabilities.append(round(fitness_scores[i] * factor, 2))
        
        added_probs.append(probabilities[0])
        for i in range(1, len(probabilities)):
            added_probs.append(added_probs[i-1] + probabilities[i])
        added_probs = [0] + added_probs

        roll = round(random.random(), 2)

        for i in range(1, len(added_probs)):
            if added_probs[i-1] <= roll <= added_probs[i]:
                return i-1
            if roll > added_probs[len(added_probs)-1]:
                return len(added_probs)-2

    
    def making_children(self, species1, species2):
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

        for i in range(len(both)):
            if both.count(both[i]) == 2:
                if both[i] not in connections_both:
                    connections_both.append(both[i])

        for i in range(len(connections_both)):
            parent_chosen.append(random.randint(0, 1))

        for i in range(len(connections_both)):
            indices = [index for index, element in enumerate(both) if element == connections_both[i]]
            copy_indexs.append(indices)

        for i in range(len(child.network)):
            for j in range(len(child.network[i])):
                for k in range(len(child.network[i][j].connections_out)):
                    if child.network[i][j].connections_out[k].enabled == True:
                        for p in range(len(connections_both)):
                            if connections_both[p][0] == child.network[i][j].number and connections_both[p][1] == child.network[i][j].connections_out[k].output_node_number:
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
