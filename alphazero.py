import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import categorical

import numpy as np
from copy import deepcopy
import math
import random

DIMENSION = 3
TEMPERATURE = 1
TEMPERATURE_THROTTLE = 6
SEARCH_LENGTH = 100
C_PUCT = 5
NOISE_CONSTANT = 0.75

FC1_DIMS = 1024
FC2_DIMS = 512
LEARNING_RATE = 0.01
DEVICE = torch.device("cpu")

MEM_SIZE = 1000000
BATCH_SIZE = 16
GAMMA = 0.95
EPISODES = 100

BATTLING_INDEX = 10
BATTLING_LENGTH = 10

class Board:
    def row_checker(self, state):
        for row in range(DIMENSION):
            total_multiplication = 1
            for column in range(DIMENSION):
                total_multiplication *= state[row][column]
            if total_multiplication == 2**DIMENSION: # Row full of twos
                return 2
            if total_multiplication == 1**DIMENSION: # Row full of ones
                return 1
        return -1
    
    def column_checker(self, state):
        for column in range(DIMENSION):
            total_multiplication = 1
            for row in range(DIMENSION):
                total_multiplication *= state[row][column]
            if total_multiplication == 2**DIMENSION: # Row full of twos
                return 2
            if total_multiplication == 1**DIMENSION: # Row full of ones
                return 1             
        return -1

    def diagonal_checker(self, state):
        for corner in range(1, 3):
            total_multiplication = 1
            if corner == 1:
                for i in range(DIMENSION):
                    total_multiplication *= state[i][i]
                if total_multiplication == 2**DIMENSION: # Row full of twos
                    return 2
                if total_multiplication == 1**DIMENSION: # Row full of ones
                    return 1
            if corner == 2:
                row = 0
                total_multiplication = 1
                for column in range(DIMENSION - 1, -1, -1):
                    total_multiplication *= state[row][column]
                    row += 1
                
                if total_multiplication == 2**DIMENSION: # Row full of twos
                    return 2
                if total_multiplication == 1**DIMENSION: # Row full of ones
                    return 1
        return -1
    
    def winning_state(self, state):
        if self.row_checker(state) != -1 or self.column_checker(state) != -1 or self.diagonal_checker(state) != -1:
            return True
        return False

    def full_board(self, state):
        zeroCounter = 0
        for row in range(DIMENSION):
            for column in range(DIMENSION):
                if state[row][column] == 0:
                    zeroCounter += 1  
        if zeroCounter == 0:
            return True
        return False
    
    def who_wins(self, state):
        if self.row_checker(state) == 1 or self.column_checker(state) == 1 or self.diagonal_checker(state) == 1:
            return 1
        if self.row_checker(state) == 2 or self.column_checker(state) == 2 or self.diagonal_checker(state) == 2:
            return -1
        if self.full_board((state)) == True:
            return 0
        return 2

    
    def who_actually_wins(self, state):
        if self.row_checker(state) == 1 or self.column_checker(state) == 1 or self.diagonal_checker(state) == 1:
            return 1
        if self.row_checker(state) == 2 or self.column_checker(state) == 2 or self.diagonal_checker(state) == 2:
            return 2
        
        return 0

    def print_formatting(self, state):
        for i in range(len(state)):
            print(state[i])


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (13, DIMENSION, DIMENSION)

        self.conv1 = nn.Conv2d(13, 2, kernel_size = 2, stride=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size = 1, stride=1)

        self.fc1 = nn.Linear(4, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)

        self.value = nn.Linear(FC2_DIMS, 1)
        self.policy = nn.Linear(FC2_DIMS, 9)


        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 1))
        x = F.relu(F.max_pool2d((self.conv2(x)), 1))

        x = x[0][0].flatten()

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy = self.policy(x)
        value = torch.tanh(self.value(x))

        return policy, value

class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0
        
        self.states = np.zeros((MEM_SIZE, *(13, 3, 3)),dtype=np.float32)
        self.outcomes = np.zeros(MEM_SIZE, dtype=np.float32)
        self.p_values = np.zeros((MEM_SIZE, *(1, 9)),dtype=np.float32)
        self.v_values = np.zeros(MEM_SIZE, dtype=np.float32)
        self.depths = np.zeros(MEM_SIZE, dtype=np.float32)
        self.p_values_selected = np.zeros(MEM_SIZE, dtype=np.float32)
    
    def add(self, state, outcome, p_value, v_value, depth, p_value_selected):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.outcomes[mem_index] = outcome
        self.p_values[mem_index] = p_value.detach()
        self.v_values[mem_index] = v_value
        self.depths[mem_index] = depth
        self.p_values_selected[mem_index] = p_value_selected

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        outcomes = self.outcomes[batch_indices]
        p_values = self.p_values[batch_indices]
        v_values = self.v_values[batch_indices]
        depths = self.depths[batch_indices]
        p_values_selected = self.p_values_selected[batch_indices]

        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        outcomes = torch.tensor(outcomes, dtype=torch.float32).to(DEVICE)
        p_values = torch.tensor(p_values, dtype=torch.float32).to(DEVICE)
        v_values = torch.tensor(v_values, dtype=torch.float32).to(DEVICE)
        depths = torch.tensor(depths, dtype=torch.float32).to(DEVICE)
        p_values_selected = torch.tensor(p_values_selected, dtype=torch.float32).to(DEVICE)

        return states, outcomes, p_values, v_values, depths, p_values_selected

class Node():
    def __init__(self, parent, state):
        self.parent = parent
        self.state = state
        self.player = None
        self.children = None

        self.value = 0
        self.visits = 0
    
    def create_children(self):  
        list_of_children = []

        for row in range(DIMENSION):
            for column in range(DIMENSION):
                if self.state[row][column] == 0:
                    temporary_state = deepcopy(self.state)
                    temporary_state[row][column] = 3 - self.player

                    temporary_node = Node(self, deepcopy(temporary_state))
                    temporary_node.player = 3 - self.player

                    list_of_children.append(temporary_node)
        
        self.children = list_of_children

    def choose_node(self, p_values):
        best_ucb = float('-inf')
        best_node = None

        for i in range(len(self.children)):
            child = self.children[i]
            if child.visits > 0:
                noise = np.random.dirichlet([1.11] * len(self.children))
                ucb = child.value/child.visits + C_PUCT * (NOISE_CONSTANT * p_values[i] + (1 - NOISE_CONSTANT) * noise[i]) * math.sqrt((math.log(self.visits))/child.visits)
            else:
                ucb = float('inf')

            if ucb > best_ucb:
                best_ucb = ucb
                best_node = child

        return best_node

class MCTS():
    def __init__(self):
        self.board = Board()
    
    def search(self, state, player, network, inner_boards_played):
        self.network = network
        self.inner_boards_played = deepcopy(inner_boards_played)

        starting_node = Node(None, state)
        starting_node.player = 3 - player
        starting_node.visits = 1
        starting_node.create_children()
        self.player = player
    
        for i in range(SEARCH_LENGTH):
            new_node = self.selection(starting_node)
            score = self.simulation(new_node)
            self.backpropogation(new_node, score)
        
        return starting_node
    
    def selection(self, node):
        while self.board.who_wins(node.state) == 2:
            if node.children == None:
                if node.visits == 0:
                    self.inner_boards_played.append(node.state)
                    return node
                
                node.create_children()
                self.inner_boards_played.append(node.state)
                return node.children[0]
            
            else:
                filler = 3 - node.player
                network_stack = self.network_prep(filler, self.inner_boards_played)
                p_values = self.network.forward(network_stack)[0]
    
                node = node.choose_node(p_values)
                self.inner_boards_played.append(node.state)
        
        return node
    
    def simulation(self, node):
        filler = 3 - node.player
        network_stack = self.network_prep(filler, self.inner_boards_played)
        score = self.network.forward(network_stack)[1]

        return score

    def backpropogation(self, node, score):
        while node.parent != None:
            node.visits += 1
            node.value += score
            node = node.parent
    
    def network_prep(self, filler, list_given):
        agent_list = deepcopy(list_given)
        opponent_list = deepcopy(list_given)
        agent_list = deepcopy(agent_list[::-1])
        opponent_list = deepcopy(opponent_list[::-1])

        for j in range(len(agent_list)):
            for r in range(DIMENSION):
                for c in range(DIMENSION):
                    if agent_list[j][r][c] == 2:
                        agent_list[j][r][c] = 0
        
        for j in range(len(opponent_list)):
            for r in range(DIMENSION):
                for c in range(DIMENSION):
                    if opponent_list[j][r][c] == 1:
                        opponent_list[j][r][c] = 0
                    if opponent_list[j][r][c] == 2:
                        opponent_list[j][r][c] = 1  
        
        network_state = []
        player = np.full((3, 3), filler)
        network_state.append(player)

        for i in range(len(agent_list)):
            if i == 6:
                break
            network_state.append(agent_list[i])
        
        if len(agent_list) != 6:
            for i in range(6 - len(agent_list)):
                network_state.append((np.zeros((3, 3))))
        
        for i in range(len(opponent_list)):
            if i == 6:
                break
            network_state.append(opponent_list[i])
        
        if len(opponent_list) != 6:
            for i in range(6 - len(opponent_list)):
                network_state.append((np.zeros((3, 3))))
            
        network_stack = np.stack(network_state)
        network_stack = torch.tensor(network_stack)
        network_stack = network_stack.unsqueeze(0)
        network_stack = network_stack.float()

        return network_stack

class Agent():
    def __init__(self):
        self.board = Board()
        self.mcts = MCTS()
        self.network = Network()
        self.memory = ReplayBuffer()
    
    def choose_training_action(self, state, player, boards_played):
        starting_node = self.mcts.search(state, player, self.network, boards_played)

        policy = float("-inf")
        for child in starting_node.children:
            latest_policy_value = child.visits**(1/TEMPERATURE)

            if latest_policy_value >= policy:
                policy = latest_policy_value
                child_chosen = child
        
        return child_chosen.state
    
    def learn(self):
        if self.memory.mem_count < BATCH_SIZE:
            return
        
        states, outcomes, p_values, v_values, depths, p_values_selected = self.memory.sample()

        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        outcomes = torch.tensor(outcomes, dtype=torch.float32).to(DEVICE)
        p_values = torch.tensor(p_values, dtype=torch.float32).to(DEVICE)
        v_values = torch.tensor(v_values, dtype=torch.float32).to(DEVICE)
        depths = torch.tensor(depths, dtype=torch.float32).to(DEVICE)
        p_values_selected = torch.tensor(p_values_selected, dtype=torch.float32).to(DEVICE)

        actual_outcome_discounted = outcomes * (GAMMA ** depths)
        value_loss = (actual_outcome_discounted - v_values) ** 2

        m = torch.distributions.Categorical(p_values[0][0])
        policy_loss = -m.log_prob(p_values_selected) * actual_outcome_discounted

        policy_loss = Variable(policy_loss, requires_grad=True)
        value_loss = Variable(value_loss, requires_grad=True)

        loss = policy_loss + value_loss
        loss = loss.mean()

        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()
    
    def save_network(self):
        self.old_network = deepcopy(self.network)
    
    def battle_old(self, state, player, boards_played):
        starting_node = self.mcts.search(state, player, self.old_network, boards_played)

        policy = float("-inf")
        for child in starting_node.children:
            latest_policy_value = child.visits**(1/TEMPERATURE)

            if latest_policy_value >= policy:
                policy = latest_policy_value
                child_chosen = child
        
        return child_chosen.state

agent = Agent()
board = Board()
boards_played = []
learning_counter = 0
agent.save_network()

for episode in range(EPISODES):
    print(episode)
    for i in range(len(boards_played)):
        print("")
        print(boards_played[i])

    state = np.zeros((DIMENSION, DIMENSION))
    index = 0
    boards_played = []
    TEMPERATURE = 1

    while board.who_wins(state) == 2:
        if index % 2 == 0:
            player = 1
        if index % 2 == 1:
            player = 2
        index += 1

        if len(boards_played) == TEMPERATURE_THROTTLE:
            TEMPERATURE = float("-inf")

        state = agent.choose_training_action(state, player, deepcopy(boards_played))
        boards_played.append(state)

    outcome = board.who_actually_wins(state)

    for i in range(len(boards_played)+1):
        if i % 2 == 0:
            filler = 1
        else:
            filler = 2

        state = agent.mcts.network_prep(filler, boards_played[:i])
        values = agent.network.forward(state)
        p_value = values[0]
        v_value = values[1]

        if filler == 1:
            p_value_selected = torch.max(p_value)
        else:
            p_value_selected = torch.min(p_value)

        agent.memory.add(state, outcome, p_value, v_value, len(boards_played)+1 - i, p_value_selected)

    agent.learn()
    learning_counter += 1

    if learning_counter % BATTLING_INDEX == 0:
        old_network_wins = 0
        new_network_wins = 0

        for e in range(BATTLING_LENGTH):
            state = np.zeros((DIMENSION, DIMENSION))
            index = 0
            boards_played = []

            while board.who_wins(state) == 2:
                if index % 2 == 0:
                    player = 1
                    if e < BATTLING_LENGTH/2:
                        state = agent.choose_training_action(state, player, deepcopy(boards_played))
                    else:
                        state = agent.battle_old(state, player, deepcopy(boards_played))

                if index % 2 == 1:
                    player = 2
                    if e < BATTLING_LENGTH/2:
                        state = agent.battle_old(state, player, deepcopy(boards_played))
                    else:
                        state = agent.choose_training_action(state, player, deepcopy(boards_played))

                index += 1
                boards_played.append(state)
            
            if e < BATTLING_LENGTH/2:
                if board.who_actually_wins(state) == 1:
                    new_network_wins += 1
                if board.who_actually_wins(state) == 2:
                    old_network_wins += 1
            else:
                if board.who_actually_wins(state) == 1:
                    old_network_wins += 1
                if board.who_actually_wins(state) == 2:
                    new_network_wins += 1
        
        if old_network_wins >= new_network_wins:
            agent.network = agent.old_network
            print("regressed")
        else:
            agent.old_network = deepcopy(agent.network)
            print("improved")




