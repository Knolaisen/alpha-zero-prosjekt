import gym 
import numpy as np

env = gym.make('CartPole-v1', render_mode="human")

#Defining constants
state_num = 4
action_num = 2
disc_size = 30
cart_pos = 4.8
cart_vel = 2*cart_pos
pole_pos =  0.418
pole_vel = 2*pole_pos
lr = 0.1
Y = 0.9

cart_pos_arr = np.linspace(-cart_pos, cart_pos, disc_size)
cart_vel_arr = np.linspace(-cart_vel, cart_vel, disc_size)
pole_pos_arr = np.linspace(-pole_pos, pole_pos, disc_size)
pole_vel_arr = np.linspace(-pole_vel, pole_vel, disc_size)

table = []

####################SHOULD MAKE TABE A LIST OF NPARRAYS"""""""""""""""""""""""""""""""

def choice(s, a):
    '''
    Returns the [L,R] "wantingness" given a state and an action
    '''
    if a == 0:
        return table[[i[0] for i in table].index(s)][1][0]
    elif a == 1:
        return table[[i[0] for i in table].index(s)][1][1]
    else:
        print("Error, invalid state or action")
        return -1

def update_func(s, a):
    '''
    Gives the q_new for a state and action
    '''
    left_obs, left_rew, left_ter, left_trun, left_info = env.step(0)
    right_obs, right_rew, right_ter, right_trun, right_info = env.step(1)
    LR = [choice(left_obs, 0), choice(right_obs, 1)]
    max_future = max(LR) 
    if max(LR) == LR[0]:
        return list(np.array(list((1-lr)*(np.array(LR)))) + np.array(lr*(left_rew+Y*max_future)))
    else:
        return list(np.array(list((1-lr)*(np.array(LR)))) + np.array(lr*(right_rew+Y*max_future)))

def updateQ(s, a):
    if s not in table:
        table.append([s, [0.5, 0.5]])
    else:
        table[[i[0] for i in table].index(s)][1] = update_func(s, a)




env.action_space.seed(42)

observation, info = env.reset(seed=42)
updateQ(observation, 0)

for i in range(1000):
    
    a = table[[i[0] for i in table].index(observation)][1]  
    print("")
    observation, reward, terminated, truncated, info = env.step(max(table[[i[0] for i in table].index(observation)][1]))

    if terminated or truncated:
        observation, info = env.reset()

env.close()