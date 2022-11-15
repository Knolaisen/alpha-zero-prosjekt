import gym
from gym_connect_four import RandomPlayer, ConnectFourEnv
env: ConnectFourEnv = gym.make("ConnectFour-v0")

player1 = RandomPlayer(env, 'Dexter-Bot')
player2 = RandomPlayer(env, 'Deedee-Bot')
#env.reset(player1, player2)
'''
Mulighet å steppe igjennom connect4 for hver spiller hver gang du skriver step.
'''
env.step(player1, player2)
env.render()
env.step(player1, player2)
env.render()
env.step(player1, player2)
env.render()
