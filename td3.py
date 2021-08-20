import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import gym
import random
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('LunarLanderContinuous-v2')
#env = gym.make('BipedalWalker-v3')

observation_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

EPISODES = 1009
LEARNING_RATE = 0.001
MEM_SIZE = 1000000
BATCH_SIZE = 256
GAMMA = 0.99
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001

FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = torch.device("cpu")

best_reward = 0
average_reward = 0
episode_number = []
average_reward_number = []

class actor_network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = env.observation_space.shape
        self.action_space = env.action_space.shape[0]

        self.fc1 = nn.Linear(*self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, self.action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x

class critic_network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_space = env.action_space.shape[0]    
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.input_shape = self.num_states + self.num_actions

        self.fc1 = nn.Linear(self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0
        
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.actions = np.zeros((MEM_SIZE, *env.action_space.shape), dtype=np.float32)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.float32).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        return states, actions, rewards, states_, dones

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

class Agent:
    def __init__(self):
        self.memory = ReplayBuffer()
        self.exploration_rate = EXPLORATION_MAX
        self.tau = 1e-2
        self.noise = OUNoise(env.action_space)
        self.time_step = 0
        self.warmup = 2000
        self.learn_step = 0
        self.update_actor = 100
        self.x = 0

        self.actor = actor_network()
        self.actor_target = actor_network()

        self.critic1 = critic_network()
        self.critic_target1 = critic_network()

        self.critic2 = critic_network()
        self.critic_target2 = critic_network()

        self.learn_step_counter = 0
        self.net_copy_interval = 10

    def choose_action(self, state, step):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0,0]
        action = self.noise.get_action(action, step)

        if self.time_step < self.warmup:
            action = env.action_space.sample()
            #action = action.detach().numpy()[0,0]
        else:
            if self.x == 0:
                print("Done")
                self.x += 1
            action = self.actor.forward(state)
            action = action.detach().numpy()[0,0]
        
        action = self.noise.get_action(action, step)
        action = action.clip(env.action_space.low, env.action_space.high)
        #print(action)
        self.time_step += 1
        return action
    
    def learn(self):
        if self.memory.mem_count < BATCH_SIZE:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample()
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        self.actor_target.eval()
        self.critic_target1.eval()
        self.critic_target2.eval()
        self.critic1.eval()
        self.critic2.eval()

        target_actions = self.actor_target.forward(states_) # need to add noise
        target_actions = target_actions + torch.clamp(torch.tensor(np.random.normal(scale=0.2)), -0.5, -0.5)
        target_actions = torch.clamp(target_actions, env.action_space.low[0], env.action_space.high[0])

        q1_ = self.critic_target1.forward(states_, target_actions)
        q2_ = self.critic_target2.forward(states_, target_actions)

        q1 = self.critic1.forward(states, actions)
        q2 = self.critic2.forward(states, actions)

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)


        critic_value_ = torch.min(q1_, q2_)
        #print(critic_value_)


        target = []
        for j in range(BATCH_SIZE):
            target.append(rewards[j] + GAMMA * critic_value_[j] * dones[j])
        
        target = torch.tensor(target).to(DEVICE)
        target = target.view(BATCH_SIZE, 1)

        """
        target = rewards + GAMMA * critic_value_ * dones
        target = torch.tensor(target).to(DEVICE)
        target = target.view(BATCH_SIZE, 1)
        """

        self.critic1.train()
        self.critic2.train()
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss

        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()
        self.critic1.eval()
        self.critic2.eval()

        self.learn_step += 1

        if self.learn_step % self.update_actor != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic1.forward(states, self.actor(states))
        actor_loss = -torch.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic1_params = self.critic1.named_parameters()
        critic2_params = self.critic2.named_parameters()
        actor_target_params = self.actor_target.named_parameters()
        critic_target1_params = self.critic_target1.named_parameters()
        critic_target2_params = self.critic_target2.named_parameters()

        critic1_state_dict = dict(critic1_params)
        critic2_state_dict = dict(critic2_params)
        actor_state_dict = dict(actor_params)
        critic_target1_dict = dict(critic_target1_params)
        critic_target2_dict = dict(critic_target2_params)
        actor_target_dict = dict(actor_target_params)

        for name in critic1_state_dict:
            critic1_state_dict[name] = tau*critic1_state_dict[name].clone() + \
                                      (1-tau)*critic_target1_dict[name].clone()
        for name in critic2_state_dict:
            critic2_state_dict[name] = tau*critic2_state_dict[name].clone() + \
                                      (1-tau)*critic_target2_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*actor_target_dict[name].clone()

        self.critic_target1.load_state_dict(critic1_state_dict)
        self.critic_target2.load_state_dict(critic2_state_dict)
        self.actor_target.load_state_dict(actor_state_dict)
    
    def returning_epsilon(self):
        return self.exploration_rate

agent = Agent()

for i in range(1, EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    score = 0
    step = 0

    while True:
        env.render()
        step += 1
        action = agent.choose_action(state, step)
        state_, reward, done, info = env.step(action)
        state_ = np.reshape(state_, [1, observation_space])
        agent.memory.add(state, action, reward, state_, done)
        agent.learn()
        state = state_
        score += reward

        if done:
            if score > best_reward:
                best_reward = score
            average_reward += score 
            print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {}".format(i, average_reward/i, best_reward, score, agent.returning_epsilon()))
            break
            
        episode_number.append(i)
        average_reward_number.append(average_reward/i)

plt.plot(episode_number, average_reward_number)
plt.show()

        

