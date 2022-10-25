import numbers
from random import random
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

import gym
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

print("environment action space", env.action_space)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    #print("Ovservation:", observation, "\nReward:", reward, "\nAction:", action)
    
    if terminated or truncated:
        observation, info = env.reset()
env.close()

numBins = 20
bins = np.array([
    np.linspace(-4.81, 4.81, numBins), # cart position
    np.linspace(-4, 4, numBins),   # cart velocity
    np.linspace(-0.418, 0.418, numBins),    # pole angle
    np.linspace(-4, 4, numBins),    # angular velocity
    ])

def discretize(observation):
    state = []
    for i in range(len(observation)):
        state.append(np.digitize(observation[i], bins[i]))
    return state

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len(observation), numBins),
            nn.ReLU(),
            nn.Linear(numBins, numBins),
            nn.ReLU(),
            nn.Linear(numBins, numBins),
            nn.ReLU(),
            nn.Linear(numBins, numBins),
            nn.ReLU(),
            nn.Linear(numBins, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        decision = np.argmax(nn.Softmax(logits))
        return decision

def policy(state): # Defining policy - takes in state and returns action
    p = random()
    if p > epsilon:
        return NeuralNetwork.forward(state)
    else:
        return env.action_space.sample()

episode_rewards = []
episode_reward = 0
learning_rate = 0.15
discount = 0.95
epsilon = 0.90

model = NeuralNetwork().to(device)
state = discretize(observation)

episode = 0
while episode < 1000:
    action = policy(state)
    observation, reward, terminated, truncated, info = env.step(action)
    memory_buffer = [state, action, reward, discretize(observation)]
    print(memory_buffer)

    episode_reward += reward
    
    
    if terminated or truncated:
        observation, info = env.reset()
        state = discretize(observation)
        print('episode_reward:', episode_reward,'\nEpsilon', epsilon, '\nepisode:', episode, '\n')
        epsilon = max(1-episode_reward/150,0.01)
        episode_rewards.append(episode_reward)
        episode_reward = 0
        episode += 1
env.close()

