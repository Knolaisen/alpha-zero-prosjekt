import gym
from gym_connect_four import RandomPlayer, ConnectFourEnv
env: ConnectFourEnv = gym.make("ConnectFour-v0")

p1 = RandomPlayer(env=env,name="P1", seed=0)
p2 = RandomPlayer(env=env, name="P2", seed=1)
#env.reset(player1, player2)
'''
Mulighet å steppe igjennom connect4 for hver spiller hver gang du skriver step.
'''

while(not env.is_win_state()):
    env.render()
    p1_move = p1.get_next_action(env)
    env._step(p1_move)
    env.render()
    p2_move = p2.get_next_action(env)
    env._step(p1_move)
    env.render()
    
