import numpy as np
from copy import deepcopy
import math
import random

DIMENSION = 3
C = 2

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

class Node():
    def __init__(self, parent, state):
        self.parent = parent
        self.state = state
        self.player = None
        self.children = None

        self.value = 0
        self.visits = 0
    
    def choose_node(self, exploration_constant):
        best_ucb = float('-inf')
        best_node = None

        for child in self.children:
            if child.visits > 0:
                ucb = child.value/child.visits + exploration_constant * math.sqrt((math.log(self.visits))/child.visits)
            else:
                ucb = float('inf')

            if ucb > best_ucb:
                best_ucb = ucb
                best_node = child

        return best_node
    
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

class MCTS:
    def __init__(self):
        self.board = Board()
        self.search_length = 1600

    def search(self, state, player):
        starting_node = Node(None, state)
        starting_node.player = 3 - player
        starting_node.visits = 1
        starting_node.create_children()
        self.player_here = player

        for i in range(self.search_length):
            new_node = self.selection(starting_node)
            score = self.simulation(new_node)
            self.backpropogation(new_node, score)
        
        best_action_value = float("-inf")

        for child in starting_node.children:
            value = child.value/child.visits
            
            if value > best_action_value:
                state = child.state
                best_action_value = value
        
        return state

    def selection(self, node):
        while self.board.who_wins(node.state) == 2:
            if node.children == None:
                if node.visits == 0:
                    return node

                node.create_children()
                return node.children[0]
            
            else:
                node = node.choose_node(2)
        
        return node
    
    def simulation(self, node):
        state = node.state
        index = 0

        if self.board.who_wins(state) != 2:
            status = self.board.who_actually_wins(state)
            if status == 3 - self.player_here:
                node.parent.value = -10000
                return status

        while self.board.who_wins(state) == 2:
            if index % 2 == 0:
                filler = 3 - node.player
            if index % 2 == 1:
                filler = node.player
            index += 1

            possible_boards = []

            for row in range(DIMENSION):
                for column in range(DIMENSION):
                    temporary_state = deepcopy(state)
                    if temporary_state[row][column] == 0:
                        temporary_state[row][column] = filler
                        possible_boards.append(deepcopy(temporary_state))
            
            state = possible_boards[random.randint(0, len(possible_boards)-1)]

        return self.board.who_actually_wins(state)
    
    def backpropogation(self, node, score):
        while node.parent != None:
            node.visits += 1

            if score == self.player_here:
                node.value += 10
            
            node = node.parent

mcts = MCTS()
board = Board()

state = np.zeros((DIMENSION, DIMENSION))
index = 0

while board.who_wins(state) == 2:
    if index % 2 == 0:
        player = 1
    if index % 2 == 1:
        player = 2
    index += 1

    state = mcts.search(state, player)
    
    print("--")
    print(state)

