import gym
from gym_connect_four import RandomPlayer, ConnectFourEnv
env: ConnectFourEnv = gym.make("ConnectFour-v0")

player1 = RandomPlayer(env, 'Dexter-Bot')
player2 = RandomPlayer(env, 'Deedee-Bot')
result = env.run(player1, player2, render=True)
reward = result.value
print(reward)