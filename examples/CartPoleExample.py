from os import stat
from turtle import st
import gym
import numpy as np
import math as mh
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import pickle


q_table = dict()
learningRate = 0.2
gamma = 0.95
epsilon_initial = 0.9
epsilon_min = 0.01
render = 5000

num_episodes = 20000
reward_history = []
epsilon_step_size = 1 / num_episodes
score_history = []

env = gym.make("CartPole-v1")


def factory():
    return [0, 0]


q_function = defaultdict(factory)


def descretize(obs):
    new_obs = [round(obs[0], 1), round(obs[1], 1), round(obs[2], 1), round(obs[3], 1)]

    return tuple(new_obs)


def discretization(o: np.ndarray):

    state0 = 0
    if o[0] < 0:
        state0 = -(mh.floor(-o[0]) + 0.5)
    else:
        state0 = mh.floor(o[0]) + 0.5

    state1 = 0
    if o[1] <= -0.5:
        state1 = -1

    elif -0.5 > o[1] < 0:
        state1 = -(mh.floor(-o[1] * 10) + 0.5) / 10

    elif 0.5 > o[1] >= 0:
        state1 = (mh.floor(o[1] * 10) + 0.5) / 10
    else:
        state1 = 1

    state2 = 0
    if o[2] < 0:
        state2 = -(mh.floor(o[2] * 10)) / 10
    else:
        state2 = mh.floor(o[2] * 10) / 10

    state3 = 0
    if o[1] <= -0.5:
        state1 = -1

    elif -0.5 > o[1] < 0:
        state1 = -(mh.floor(-o[1] * 10) + 0.5) / 10

    elif 0.5 > o[1] >= 0:
        state1 = (mh.floor(o[1] * 10) + 0.5) / 10
    else:
        state1 = 1

    return (float(state0), float(state1), float(state2), float(state3))


def update_epsilon(epsilon):
    return max(epsilon_min, min(epsilon_initial, epsilon - epsilon_step_size))


epsilon = epsilon_initial
for i in range(num_episodes):
    state = env.reset()
    state = descretize(state)

    done = False
    score = 0

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            if q_function[state][0] == q_function[state][1]:
                action = env.action_space.sample()

            else:
                action = np.argmax(q_function[state])

        next_state, reward, done, _ = env.step(action)
        next_state = descretize(next_state)

        epsilon = update_epsilon(epsilon)

        # Update q_function
        q_function[state][action] = q_function[state][action] + learningRate * (
            reward + gamma + max(q_function[next_state]) - q_function[state][action]
        )

        state = next_state
        score += reward
        if i % render == 0:
            env.render()

    reward_history.append(score)


with open("policy.pi", "wb") as f:
    pickle.dump(q_function, f)


for i in range(1000):
    state = env.reset()
    state = descretize(state)

    done = False
    score = 0

    while not done:

        action = np.argmax(q_function[state])

        next_state, reward, done, _ = env.step(action)

        state = descretize(next_state)
        score += reward
        if i % 500 == 0:
            env.render()

    score_history.append(score)


print(max(score_history))
# plt.plot(score_history)
# plt.show()
