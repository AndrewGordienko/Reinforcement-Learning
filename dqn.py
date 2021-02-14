import gym
import keras
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from collections import deque
from matplotlib import pyplot as plt

# we want to have our network output a number one or two
# then play the number 
# every so often retrain the network
# make a memory where u save the stuff so u can retrain 

env = gym.make('CartPole-v0')
episodes = 10000
exploration_max = 1.0
learningRate = 0.0001
memorySize = 1000000
batchSize = 20
gamma = 0.95
explorationDecay = 0.995
explorationMin = 0.01

class DQN_Solver:

    def __init__(self, action_Space, env, observation_space):
        self.actionSpace = action_space
        self.exploration_rate = exploration_max
        self.memory = deque(maxlen=memorySize)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space, ), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(action_Space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=learningRate))

    def remember(self, state, action, reward, nextState, terminal):
        self.memory.append((state, action, reward, nextState, terminal))

    def act():
        action = env.action_space.sample()
        return action

    def newAct(self, state):
        ourNumber = random.random()
        if ourNumber < self.exploration_rate:
            return random.randrange(self.actionSpace)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def retrain(self):
        if len(self.memory) < batchSize:
            return
        batch = random.sample(self.memory, batchSize)
        for state, action, reward, nextState, terminal in batch:
            qUpdate = reward
            if not terminal:
                qUpdate = (reward + gamma * np.amax(self.model.predict(nextState)[0]))
            
            qValues = self.model.predict(state)
            qValues[0][action] = qUpdate
            self.model.fit(state, qValues, verbose = 0)

        self.exploration_rate *= explorationDecay
        self.exploration_rate = max(explorationMin, self.exploration_rate)

    def save(self):
        self.model.save('C:/Users/andrew/Desktop/All Code/Eve')

    def takeOut(self):
        self.model = keras.models.load_model('C:/Users/andrew/Desktop/All Code/Eve')

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n


Solver = DQN_Solver(action_space, env, observation_space)

bestScore = 0
i = 0
lastReward = 0

for q in range(250):
    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    step = 0
    score = 0
    i += 1
    while True:
        env.render()
        #print(state)
        step += 1
        action = Solver.newAct(state)
        passState, reward, terminal, info = env.step(action)
        reward = reward if not terminal else -reward
        nextState = np.reshape(passState, [1, observation_space])
        Solver.remember(state, action, reward, nextState, terminal)
        state = nextState
        Solver.retrain()
        score += reward
        """
        state = np.reshape(state, [1, observation_space])
        action = Solver.newAct(state)
        nextState, reward, terminal, info = env.step(action)
        reward = reward if not terminal else -reward
        nextState = np.reshape(nextState, [1, observation_space])
        Solver.remember(state, action, reward, nextState, terminal)
        state = nextState
        """

        if terminal:
            if reward > lastReward:
                Solver.save()
                Solver.takeOut()
                lastReward = reward
            bestScore = max(step, bestScore)

            """
            if bestScore < step:
                Solver.save()
                Solver.takeOut()
            bestScore = max(step, bestScore)
            """
            print("Episode {} High Score {} Last Score {} Reward {}".format(i, bestScore, step, score))
            break

                



env.close()
