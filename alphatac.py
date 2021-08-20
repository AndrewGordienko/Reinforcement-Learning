# working on alphazero for tic tac toe

import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from copy import copy, deepcopy
from torch.distributions import categorical

DIMENSION = 3
EMPTY_TABLE = np.zeros((DIMENSION, DIMENSION))
OBSERVATION_SPACE = (DIMENSION, DIMENSION)
ACTION_SPACE = (DIMENSION, DIMENSION)
PATH = "./tictactoemodel"

FC1_DIMS = 3
FC2_DIMS = 9
DEVICE = torch.device("cpu")
LEARNING_RATE = 0.01
MEM_SIZE = 1000000
EPSILON = 0.9

SELECTION_DEPTH = 6
SIMULATION_DEPTH = 4
EPISODES = 10001

EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.01
GAMMA = 0.95
BATCH_SIZE = 64
C_PUCT = 2
episode_changing_number = 100

TESTING_GAME_NUMBER = 100
TRAINING_LOOPS_NUMBER = 100

EXPLORATION_CONSTANT = 1

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
    
    def who_wins(self, state, filler):
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
        self.input_shape = (13, 3, 3)

        self.conv1 = nn.Conv2d(13, 2, kernel_size = 2, stride=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size = 1, stride=1)

        self.fc1 = nn.Linear(4, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.value = nn.Linear(512, 1)
        self.policy = nn.Linear(512, 9)


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


class ReplayBuffer: # player one replay buffer
    def __init__(self):
        self.mem_count = 0
        
        self.states = np.zeros((MEM_SIZE, *(13, 3, 3)),dtype=np.float32)
        self.outcomes = np.zeros(MEM_SIZE, dtype=np.float32)
        self.p_values = np.zeros((MEM_SIZE, *(1, 9)),dtype=np.float32)
        self.v_values = np.zeros(MEM_SIZE, dtype=np.float32)
        self.depths = np.zeros(MEM_SIZE, dtype=np.float32)
        self.p_values_taken = np.zeros(MEM_SIZE, dtype=np.float32)

    
    def add(self, state, outcome, p_value, v_value, depth, p_value_taken):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.outcomes[mem_index] = outcome
        self.p_values[mem_index] = (p_value.detach())
        self.v_values[mem_index] = v_value
        self.depths[mem_index] = depth
        self.p_values_taken[mem_index] = p_value_taken

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        outcomes = self.outcomes[batch_indices]
        p_values = self.p_values[batch_indices]
        v_values = self.v_values[batch_indices]
        depths = self.depths[batch_indices]
        p_values_taken = self.p_values_taken[batch_indices]

        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        outcomes = torch.tensor(outcomes, dtype=torch.float32).to(DEVICE)
        p_values = torch.tensor(p_values, dtype=torch.float32).to(DEVICE)
        v_values = torch.tensor(v_values, dtype=torch.float32).to(DEVICE)
        depths = torch.tensor(depths, dtype=torch.float32).to(DEVICE)
        p_values_taken = torch.tensor(p_values_taken, dtype=torch.float32).to(DEVICE)
        

        return states, outcomes, p_values, v_values, depths, p_values_taken

class Node():
    def __init__(self):
        self.parent = None
        self.children = None
        self.state = None
        self.n = 0
        self.w = 0
        self.token = 0
        self.p = 0
    
class MCTS():
    def __init__(self):
        self.iteration_number = 100
        self.bug = 0
        self.board = Board()
        self.network = Network()
        self.replay_buffer = ReplayBuffer()
    
    def make_children(self, state, filler, parent_node, boards_played):
        children_list = []

        potential_moves = 0
        for row in range(DIMENSION):
            for column in range(DIMENSION):
                if state[row][column] == 0:
                    potential_moves += 1
        
        s = np.random.dirichlet([1.11] * potential_moves) # this might always be one 

        for row in range(DIMENSION):
            for column in range(DIMENSION):
                temp_state = deepcopy(state)
                if temp_state[row][column] == 0:
                    temp_state[row][column] = filler

                    node = Node()
                    node.state = deepcopy(temp_state)
                    node.token = filler
                    node.parent = parent_node

                    network_stack = self.network_prep(filler, boards_played)
                    policy, value = self.network(network_stack)

                    another_temp_state = deepcopy(parent_node.state)
                        
                    another_temp_state = another_temp_state.flatten()

                    for k in range(len(policy)):
                        if another_temp_state[k] == 0:
                            another_temp_state[k] = filler
                            if str(another_temp_state) == str(temp_state.flatten()):
                                p = policy[k]

                                index_picked = random.randint(0, len(s) - 1)
                                s_picked = s[index_picked]

                                node.p = 0.5 * p + (1 - 0.5) * s_picked

                                s = np.delete(s, index_picked)

                            
                            another_temp_state[k] = 0



                    children_list.append((node))
        

        return children_list
    
    def move_sum(self, node_given):
        move_sum = 0

        children = node_given.children

        for i in range(len(children)):
            move_sum += children[i].n

        return move_sum
    
    def max_move_number(self, node_given):
        number = float("-inf")

        children = node_given.children

        for i in range(len(children)):
            if children[i].n > number:
                node_picked = children[i]
                number = children[i].n
        
        return node_picked
    
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

    def fill_identifier(self, filler, i):
        if i % 2 == 0:
            if filler == 1:
                return 1, 2
            else:
                return 2, 1

        else:
            if filler == 1:
                return 2, 1
            else:
                return 1, 2

    def play_move(self, node):
        value = float("-inf")
        sum = self.move_sum(node)

        for i in range(len(node.children)):
            policy = node.children[i].n**(1/self.EXPLORATION_CONSTANT)/sum**(1/self.EXPLORATION_CONSTANT)

            if policy >= value:
                value = policy
                node_picked = node.children[i]

        return node_picked
    
    def competitive_move(self, node):
        value = float("-inf")

        for i in range(len(node.children)):
            if node.children[i].n >= value:
                value = node.children[i].n
                node_picked = node.children[i]
        
        return node_picked

    def main(self, starting_state, token, list_of_boards, indicator):
        if token == 1:
            temp_filler = 2
        if token == 2:
            temp_filler = 1

        start_node = Node()
        start_node.state = starting_state
        start_node.token = temp_filler
        boards_played = []
        boards_played.append(list_of_boards)


        for i in range(self.iteration_number):
            node = start_node
            state = starting_state
            
            another_i = -1
            boards_played = []

            if len(boards_played) + len(list_of_boards) <= 4:
                self.EXPLORATION_CONSTANT = 1
            else:
                self.EXPLORATION_CONSTANT = float("-inf")

            while True:
                state = node.state
                another_i += 1

                if another_i % 2 == 0:
                    filler = token
                else:
                    if token == 1:
                        filler = 2
                    if token == 2:
                        filler = 1
                
                if node.children != None:
                    
                    network_stack = self.network_prep(filler, boards_played)
                    policy, value = self.network(network_stack)

                    if node.token == 1:
                        number_to_beat = float("-inf")
                    else:
                        number_to_beat = float("inf")

                    for n in range(len(node.children)):
                        sum = self.move_sum(node)
                        q = node.children[n].w/(node.children[n].n + 1)

                        p = node.children[n].p

                        u = C_PUCT * p * math.sqrt(sum)/(1 + node.children[n].n)

                        total = q + u

                        if node.token == 1:
                            if total > number_to_beat:
                                node_picked = node.children[n]
                                number_to_beat = total
                        else:
                            if total < number_to_beat:
                                node_picked = node.children[n]
                                number_to_beat = total
                    
                    boards_played.append(node_picked.state)
                    node = node_picked
                
                else:
                    #print("child")
                    node.children = self.make_children(node.state, filler, node, boards_played)
                    node_picked = node.children[random.randrange(0, len(node.children))] # move picked from expansion
                    boards_played.append(node_picked.state)
                    node = node_picked
                    # add backprop
                    break

                
                if self.board.who_wins(node.state, filler) != 2:
                    
                    # board state is done backpropogate
                    #print(boards_played)

                    # time for backprop
                    #print(node.state)
                    outcome = board.who_wins(node.state, 1)
                    temp_node = node

                    network_stack = self.network_prep(token, boards_played)
                    policy1, value1 = self.network(network_stack)

                    copy_list = deepcopy(boards_played)
                    l = 0

                    #print("policy")

                    while temp_node.parent != None:
                        if (len(boards_played) - l - 1) % 2 == 0:
                            filler = token
                        else:
                            if token == 1:
                                filler = 2
                            if token == 2:
                                filler = 1

                        temp_node.n += 1
                        temp_node.w += value1
                        temp_node = temp_node.parent
                        
                        copy_list = deepcopy(boards_played)
                        #del copy_list[len(boards_played) - l]

                        for u in range(l):
                            copy_list.pop()

                        temp_copy = deepcopy(copy_list)

                        network_stack = self.network_prep(token, temp_copy)
                        policy, value = self.network(network_stack)

                        #print("")
                        #print(copy_list)
                        #print(policy)


                        self.replay_buffer.add(temp_node.state, outcome, policy, value1, (len(boards_played) - l), temp_node.p)

                        l += 1

                    break
                            
            
            



            
            # maybe just pull the backprop out to here

        #print("verdict")
        #print((self.max_move_number(start_node).state))
        if indicator == 1:
            return (self.play_move(start_node).state)
        else:
            return (self.competitive_move(start_node).state)
        #return (self.max_move_number(start_node).state)






    def retrain(self):
        if self.replay_buffer.mem_count < BATCH_SIZE:
            return
        
        print("train")
        self.bug = 1
        self.old_network = deepcopy(self.network)
        
        states, outcomes, p_values, v_values, depths, p_values_taken = self.replay_buffer.sample()
        actual_outcome_discounted = outcomes * (GAMMA ** depths)

        """
        print("values")
        print(p_values_taken)
        print("number")
        print(actual_outcome_discounted)
        print("more")
        print(p_values)
        """
        
        #print(new_list)

 

        value_loss = ((actual_outcome_discounted - v_values) ** 2)
        m = torch.distributions.Categorical(p_values)


        
        #print("here")
        #print(p_values_taken)


        policy_loss = (-m.log_prob(p_values_taken) * actual_outcome_discounted)
        #policy_loss = (-m.log_prob(p_values_taken))


        policy_loss = Variable(policy_loss, requires_grad=True)
        value_loss = Variable(value_loss, requires_grad=True)

        loss = policy_loss + value_loss
        loss = loss.mean()
        #print(loss)

        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.safe_network = deepcopy(self.network)


    
    def add_memory(self, state, outcome, p_value, v_value, depth, p_value_taken):
        self.replay_buffer.add(state, outcome, p_value, v_value, depth, p_value_taken)


    def network_used(self, token):
        if token == 1:
            self.network = deepcopy(self.old_network)
        if token == 2:
            self.network = deepcopy(self.safe_network)
    
    def update_positive(self):
        self.network = deepcopy(self.safe_network)
    
    def update_negative(self):
        self.network = deepcopy(self.old_network)
    
    def saving_network(self):
        torch.save(self.network.state_dict(), PATH)
        print("saved")
    
    def loading_network(self):
        self.network.load_state_dict(torch.load(PATH))
        print("loaded")
    
    def printing_values(self, boards_played, filler):
        network_stack = self.network_prep(filler, boards_played)
        policy, value = self.network(network_stack)

        print(random.randint(0, 10))

        return policy, value


list_of_boards = []
mcts = MCTS()
#mcts.main(np.zeros((3, 3)), 1, list_of_boards)


board = Board()
player_1_wins, player_2_wins, ties = 0, 0, 0

#mcts.saving_network()


mcts.loading_network()

for i in range(EPISODES):
    print(i)
    state = np.zeros((3, 3))
    list_of_boards = []

    while True:
        if board.winning_state(state) == False and board.full_board(state) == False:
            state = mcts.main(state, 1, list_of_boards, 1)
            list_of_boards.append(state)

        else:
            break

        if board.winning_state(state) == False and board.full_board(state) == False:
            state = mcts.main(state, 2, list_of_boards, 1)
            list_of_boards.append(state)
        else:
            break
    
    if board.who_actually_wins(state) == 1:
        player_1_wins += 1
    if board.who_actually_wins(state) == 2:
        player_2_wins += 1
    if board.who_actually_wins(state) == 0:
        ties += 1
    
    #mcts.add_memory(state, outcome, p_value, v_value, depth, p_value_taken)

    if i % 100 == 0:
        print(player_1_wins, player_2_wins, ties, i)
        player_1_wins, player_2_wins, ties = 0, 0, 0

        for n in range(len(list_of_boards)):
            print("")
            print(list_of_boards[n])


    
    if i % TRAINING_LOOPS_NUMBER == 0:
        mcts.retrain()
        if mcts.bug == 1:
            print("attempt")
            player_1_wins, player_2_wins, ties = 0, 0, 0
            my_network_token_1, my_network_token_2 = 1, 2

            for n in range(TESTING_GAME_NUMBER):
                list_of_boards = []
                state = np.zeros((3, 3))

                if n % (TESTING_GAME_NUMBER/2) == 0 and n != 0:
                    my_network_token_1, my_network_token_2 = 2, 1
                    compare_1_wins, compare_2_wins, compare_ties = deepcopy(player_1_wins), deepcopy(player_2_wins), deepcopy(ties)
                    player_1_wins, player_2_wins, ties = 0, 0, 0


                while True:
                    if board.winning_state(state) == False and board.full_board(state) == False:
                        mcts.network_used(my_network_token_1)
                        state = mcts.main(state, 1, list_of_boards, 2)
                        list_of_boards.append(state)

                    else:
                        break

                    if board.winning_state(state) == False and board.full_board(state) == False:
                        mcts.network_used(my_network_token_2)
                        state = mcts.main(state, 2, list_of_boards, 2)
                        list_of_boards.append(state)

                    else:
                        break
                
                print("episode {} {}".format(n, board.who_actually_wins(state)))

                if board.who_actually_wins(state) == 1:

                    player_1_wins += 1
                if board.who_actually_wins(state) == 2:
                    player_2_wins += 1
                if board.who_actually_wins(state) == 0:
                    ties += 1
            
            if compare_1_wins + player_2_wins + compare_2_wins + player_1_wins != 0:
                here = 100/(compare_1_wins + player_2_wins + compare_2_wins + player_1_wins)
            else:
                here = 1
            
            adjusted1 = here * (compare_1_wins + player_2_wins)
            adjusted2 = here * (compare_2_wins + player_1_wins)
            print(adjusted1, adjusted2)
            if adjusted2 >= 50:
                print("adjusted")
                mcts.update_positive()
                mcts.saving_network()
                mcts.loading_network()

                for w in range(len(list_of_boards)):
                    print("")
                    print(list_of_boards[w])

            else:
                mcts.update_negative()
            
            player_1_wins, player_2_wins, ties = 0, 0, 0
        


while True:
    state = np.zeros((3, 3))
    turn = random.randint(0, 1)
    if turn == 0:
        hard_turn_1 = 1
        hard_turn_2 = 2
    if turn == 1:
        hard_turn_1 = 2
        hard_turn_2 = 1
    list_of_boards = []
    print("Lets play a game!")

    while board.winning_state(state) == False and board.full_board(state) == False:
        print("")
        print(board.print_formatting(state))

        if turn % 2 == 0:
            state = mcts.main(state, hard_turn_1, list_of_boards, 2)
            print(mcts.printing_values(list_of_boards, hard_turn_1))
            list_of_boards.append((state))


        else:
            # human plays
            while True:
                print("-----")
                y = int(input("Please enter row "))
                x = int(input("Please enter column "))
                print("-----")

                if state[y][x] == 0:
                    state[y][x] = hard_turn_2
                    list_of_boards.append((state))
                    break

    
        turn += 1
    
    print(board.print_formatting(state))

