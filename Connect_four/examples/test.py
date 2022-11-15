import gym
import gym_connect_four

env = gym.make("ConnectFour-v0")
env.render()

print(env.available_moves())
env.step(2)

env.render()
env.step(2)
env.render()
