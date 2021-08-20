import random
import numpy as np

dimention = 3
emptyTable = np.zeros((dimention, dimention))

def printFormmating(boardState):
    for i in range(len(boardState)):
        print(boardState[i])

def rowChecker(boardState, dimention):
    for row in range(dimention):
        totalMultiplication = 1
        for column in range(dimention):
            totalMultiplication *= boardState[row][column]

        if totalMultiplication == 2**dimention: # Row full of twos
            return 2
        
        if totalMultiplication == 1**dimention: # Row full of ones
            return 1
        
    return -1

def columnChecker(boardState, dimention):
    for column in range(dimention):
        totalMultiplication = 1
        for row in range(dimention):
            totalMultiplication *= boardState[row][column]

        if totalMultiplication == 2**dimention: # Row full of twos
            return 2
        
        if totalMultiplication == 1**dimention: # Row full of ones
            return 1
        
    return -1


def diagonalChecker(boardState, dimention):
    for corner in range(1, 3):
        totalMultiplication = 1

        if corner == 1:
            for i in range(dimention):
                totalMultiplication *= boardState[i][i]

            
            if totalMultiplication == 2**dimention: # Row full of twos
                return 2
        
            if totalMultiplication == 1**dimention: # Row full of ones
                return 1
        
        if corner == 2:
            row = 0
            totalMultiplication = 1
            for column in range(dimention - 1, -1, -1):
                totalMultiplication *= boardState[row][column]
                row += 1
            
            if totalMultiplication == 2**dimention: # Row full of twos
                return 2
        
            if totalMultiplication == 1**dimention: # Row full of ones
                return 1
    
    return -1
                
            
def winningState(boardState, dimention):
    if rowChecker(boardState, dimention) != -1 or columnChecker(boardState, dimention) != -1 or diagonalChecker(boardState, dimention) != -1:
        return True
    
    return False

def whoWins(boardState, dimention):
    if rowChecker(boardState, dimention) == 1 or columnChecker(boardState, dimention) == 1 or diagonalChecker(boardState, dimention) == 1:
        return -1
    
    if rowChecker(boardState, dimention) == 2 or columnChecker(boardState, dimention) == 2 or diagonalChecker(boardState, dimention) == 2:
        return 1
    
    if fullBoard(boardState, dimention) == True:
        return 0
    
    return 3



def fullBoard(boardState, dimention):
    zeroCounter = 0
    for row in range(dimention):
        for column in range(dimention):
            if boardState[row][column] == 0:
                zeroCounter += 1
    
    if zeroCounter == 0:
        return True
    
    return False


def bestMove(boardState):
    bestScore = -100000
    for row in range(dimention):
        for column in range(dimention):
            if boardState[row][column] == 0:
                boardState[row][column] = 2
                score = minimax(boardState, 0, -100000, 100000, False)
                boardState[row][column] = 0
                if bestScore < score:
                    bestScore = score
                    move = row, column
    
    
    return move

def minimax(boardState, depth, alpha, beta, isMaximizing):

    if winningState(boardState, dimention) == True:
      return(whoWins(boardState, dimention))
      
    if fullBoard(boardState, dimention) == True:
      return 0
    
    if isMaximizing:
        # ai looking for its best move (the highest number for the board)
        bestScore = -100000
        for row in range(dimention):
            for column in range(dimention):
                if boardState[row][column] == 0:
                    boardState[row][column] = 2
                    score = minimax(boardState, depth + 1, alpha, beta, False)
                    alpha = max(alpha, score)
                    boardState[row][column] = 0
                    bestScore = max(score, bestScore)
                    if beta <= alpha:
                        break
                    
        return bestScore
    
    if not isMaximizing:
        bestScore = 100000
        # huamn will pick its best move (low reward for us)
        for row in range(dimention):
            for column in range(dimention):
                if boardState[row][column] == 0:
                    boardState[row][column] = 1
                    score = minimax(boardState, depth + 1, alpha, beta, True)
                    beta = min(beta, score)
                    boardState[row][column] = 0
                    bestScore = min(score, bestScore)
                    if beta <= alpha:
                        break


        return bestScore


    


while True:
    boardState = emptyTable
    turn = random.randint(1, 2)
    print("Lets play a game!")

    while fullBoard(boardState, dimention) == False and winningState(boardState, dimention) == False:
        print("")
        print(printFormmating(boardState))

        if turn % 2 == 0:
            # Ai play
            y, x = (bestMove(boardState))
            boardState[y][x] = 2
        
        else:
            # human plays
            while True:
                print("-----")
                y = int(input("Please enter row "))
                x = int(input("Please enter column "))
                print("-----")

                if boardState[y][x] == 0:
                    boardState[y][x] = 1
                    break
    
        turn += 1
    
    print(printFormmating(boardState))

    if fullBoard(boardState, dimention) == True:
        print("Board is full")
        break
        
    
    if winningState(boardState, dimention) == True:
        print("{} won".format(whoWins(boardState, dimention)))
        break
        
    