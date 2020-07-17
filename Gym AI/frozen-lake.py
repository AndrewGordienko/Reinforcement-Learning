import numpy as np
import gym
import random
import time
from IPython.core.display import clear_output

env = gym.make("FrozenLake-v0")

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))

num_episode = 10000
max_steps = 100

learning_rate = 0.8
gamma = 0.95
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

all_rewards = []
episode = 1

while episode != num_episode:
    state = env.reset()

    done = False
    rewards_rn = 0

    for step in range(max_steps):
        # What will the gods of chance command us to do?
        command = random.uniform(0, 1)

        if command > exploration_rate:
            # Be greedy!!!
            action = np.argmax(q_table[state, :])
        else:
            # Be curious O_o
            action = env.action_space.sample()

        # Let us march forwards!
        new_state, reward, done, info = env.step(action)

        # Update with a lot of fancy math
        q_table[state, action] = q_table[state, action] + learning_rate * (
        reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])

        state = new_state
        rewards_rn += reward

        if done:
            break

    # We must clip our wings and be less curious :(
    exploration_rate = min_exploration_rate + \
                       (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    # Are we rich yet?
    all_rewards.append(rewards_rn)
    episode += 1

    # Let's print our success, or failure
    if episode % 1000 == 0:
        print(episode, (sum(all_rewards))/1000)
        all_rewards = []

# Lets watch it in action! Will we be victorious?
for episode in range(3):
    state = env.reset()
    done = False
    print("Episode ", episode + 1)
    time.sleep(1)

    for step in range(max_steps):
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)

        action = np.argmax(q_table[state, :])  # Where the magic happens
        new_state, reward, done, info = env.step(action)

        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print("Frisbee Collected!")
                time.sleep(3)
            else:
                print("Its a cold world down there isn't it")
                time.sleep(3)

            clear_output(wait=True)
            break

        state = new_state

env.close()
