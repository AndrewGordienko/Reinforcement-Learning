import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import gym
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1').unwrapped
#env = gym.make('LunarLander-v2')
#env = gym.make('LunarLander-v2')

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

episodes = 1001
exploration_max = 1.0
learningRate = 0.0001
mem_size = 1000000
batchSize = 64
gamma = 0.99
explorationDecay = 0.9999
explorationMin = 0.01

totalSteps = 0
step = 0
bestReward = 0
averageReward = 0

class ReplayBuffer:
    def __init__(self):
        self.mem_size = mem_size
        self.mem_count = 0

        self.actions = np.zeros( self.mem_size, dtype=np.int64)
        self.rewards = np.zeros( self.mem_size, dtype=np.float32)
    
    def add(self, action, reward):
        mem_index = self.mem_count % self.mem_size 
        
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward

        self.mem_count += 1
    
    def sample(self):
        mem_max = min(self.mem_count, self.mem_size)
        batch_indices = np.random.choice(mem_max, batchSize, replace=True)

        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]

        return actions, rewards
    
    def reset_buffer(self):
        self.mem_size = mem_size
        self.mem_count = 0

        self.actions = np.zeros( self.mem_size, dtype=np.int64)
        self.rewards = np.zeros( self.mem_size, dtype=np.float32)

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = env.observation_space.shape
        self.action_space = action_space

        self.fc1Dims = 1024
        self.fc2Dims = 512

        self.fc1 = nn.Linear(*self.input_shape, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)
        self.fc3 = nn.Linear(self.fc2Dims, self.action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)
        self.device = torch.device("cpu")
        self.to(self.device)
    
    def forward(self, state):
        state = torch.Tensor(state).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Agent:
    def __init__(self, exploration_max, action_space):
        self.epsilon = exploration_max
        self.action_space = action_space
        self.memory = ReplayBuffer()

        self.policy = Network()
        self.policy.device = 'cpu'


    def choose_action(self, state):
        probabilities = F.softmax(self.policy.forward(state))
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)

        return action.item(), log_probs
    
    def return_epsilon(self):
        return self.epsilon

    def learn(self):
        self.policy.optimizer.zero_grad()

        actions, rewards = self.memory.sample()
        actions = torch.tensor(actions, dtype=torch.long).to(self.policy.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.policy.device)

        batch_indices = np.arange(batchSize, dtype=np.int64)
        G = np.zeros_like(rewards, dtype=np.float64)

        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k] * discount
                discount *= gamma
            G[t] = G_sum
        
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std

        G = torch.tensor(G, dtype=torch.float32).to(self.policy.device)

        loss = 0
        for g, logprob in zip(G, actions):
            loss += -g * logprob

        loss = Variable(loss, requires_grad=True)
        
        loss.backward()
        self.policy.optimizer.step()

        self.memory.reset_buffer()






        

agent = Agent(exploration_max, action_space)
episodeNumber = []
averageRewardNumber = []

for i in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, observation_space])

    score = 0
    done = False
    while not done:
        step += 1
        env.render()
        action, log_probs = agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        state_ = np.reshape(state_, [1, observation_space])
        agent.memory.add(log_probs, reward)
        state = state_

        #if done: reward = -reward
        score += reward

        agent.learn()
    
    if score > bestReward:
        bestReward = score
    averageReward += score

    print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {}".format(i, averageReward/(i+1), bestReward, score, agent.return_epsilon()))

    
    episodeNumber.append(i)
    averageRewardNumber.append(averageReward/(i+1))

plt.plot(episodeNumber, averageRewardNumber)
plt.show()


