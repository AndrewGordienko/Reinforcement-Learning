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
batchSize = 32
gamma = 0.95
explorationDecay = 0.995
explorationMin = 0.01
tau = 0.1

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

def createNetwork():
    model = Sequential()
    model.add(Dense(24, input_shape=(observation_space, ), activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(action_space, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=learningRate))

    return model

class DQN_Solver:

    def __init__(self, action_Space, env, observation_space):
        self.actionSpace = action_space
        self.exploration_rate = exploration_max
        self.memory = deque(maxlen=memorySize)

        self.actionNetwork = createNetwork()
        self.targetNetwork = createNetwork()



    def remember(self, state, action, reward, nextState, terminal):
        self.memory.append((state, action, reward, nextState, terminal))


    def newAct(self, state):
        ourNumber = random.random()
        if ourNumber < self.exploration_rate:
            return random.randrange(self.actionSpace)

        q_values = self.actionNetwork.predict(state)
        return np.argmax(q_values[0])

    def retrain(self):
        # also known as replay buffer
        minibatchValues = []
        if len(self.memory) < batchSize:
            return

        batch = random.sample(self.memory, batchSize)
        for state, action, reward, nextState, terminal in batch:

            experienceNewValues = self.actionNetwork.predict(state)[0]
            if terminal:
                qUpdate = reward
            else:
                # using online network to select an action
                actionNetworkChoice = np.argmax(self.actionNetwork.predict(state)[0])
                # using the target network to select
                targetNetworkChoice = np.argmax(self.targetNetwork.predict(state)[0][actionNetworkChoice])
                qUpdate = (reward + gamma * targetNetworkChoice)
            
            experienceNewValues[action] = qUpdate
            minibatchValues.append(experienceNewValues)

        miniBatchStates = np.array([e[0] for e in batch])
        miniBatchNewValues = np.array(minibatchValues)

        self.actionNetwork.fit(miniBatchStates, miniBatchNewValues, verbose=0)
        self.exploration_rate *= explorationDecay
        self.exploration_rate = max(explorationMin, self.exploration_rate)

    def updateTarget(self):
        self.actionNetworkTheta = self.actionNetwork.get_weights()
        self.targetNetworkTheta = self.targetNetwork.get_weights()
        counter = 0
        for q_weight, target_weight in zip(self.actionNetworkTheta, self.targetNetworkTheta):
            target_weight = target_weight * (1-tau) + q_weight * tau
            self.targetNetworkTheta[counter] = target_weight
            counter += 1

        self.targetNetwork.set_weights(self.targetNetworkTheta)

        



           

    def save(self):
        self.model.save('C:/Users/andrew/Desktop/All Code/Eve')

    def takeOut(self):
        self.model = keras.models.load_model('C:/Users/andrew/Desktop/All Code/Eve')




Solver = DQN_Solver(action_space, env, observation_space)

bestScore = 0
i = 0
lastReward = 0
totalSteps = 0
step = 0

while True:
    totalSteps += step
    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    step = 0
    score = 0
    i += 1

    if totalSteps % 10000:
        Solver.updateTarget()
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
        
        if terminal:
            if reward > lastReward:
                Solver.save()
                Solver.takeOut()
                lastReward = reward
            bestScore = max(step, bestScore)

            
            print("Episode {} High Score {} Last Score {} Reward {}".format(i, bestScore, step, score))
            break

                



env.close()
