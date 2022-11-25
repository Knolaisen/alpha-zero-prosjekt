import env
import numpy as np
from neural_network import model
import pickle

# Dictionary storing all visited board states and their repective empirical value and visit count.
filename = 'nodes'
try:
	infile = open(filename, 'rb')
	node_dictionary = pickle.load(infile)
	infile.close()
except (OSError, IOError) as e:
	node_dictionary = {}

def simulate(board, action, player) -> float:
	
	simulation_game = env.ConnectFour()
	simulation_game.board = board
	#print(f'\nSimulating game starting with player {player} on the following board')
	#print(simulation_game.board)
	#print(f'Simulating move at column {action}')
	simulation_game.move(column=action, player=player)
	
	start_id = hash(simulation_game.board.data.tobytes())
	id_buffer = [] # keep track on board states visited
	id_buffer.append(start_id)
	
	if start_id in node_dictionary:
		node_dictionary[start_id][1] += 1 # Adding 1 on number of visits at this state
	else:	
		node_dictionary[start_id] = [0,1] # Making a dictionary entry starting with 1 number of visits
	
	start = True
	while not simulation_game.is_game_over():
		# Enter if board state has been visited before. Dictionary values: [wins, visits]
		id = hash(simulation_game.board.data.tobytes())
		if not start:
			id_buffer.append(id)
			if id in node_dictionary:
				node_dictionary[id][1] += 1 # Adding 1 on number of visits at this state
			else:
				node_dictionary[id] = [0,1] # Making a dictionary entry starting with 1 number of visits
			
		#policy = np.random.rand(7)
		policy, value = model.forward(simulation_game.board)
		policy = simulation_game.trim_and_normalize(policy)
		action = select(policy)
		simulation_game.move(action, player)
		player *= -1
		start = False
		
	reward = simulation_game.is_win_state()[1]
	#print(f'\nThe simulated winner is {reward} with board \n{simulation_game.board}')
	 
	for id in id_buffer:
		node_dictionary[id][0] += reward
		reward *= -1
	empirical_value, number_of_visits = node_dictionary[id_buffer[0]]
	empirical_mean_value = empirical_value/number_of_visits
	
	return empirical_mean_value

def select(policy): #TODO include UBC & value and make sure it's correctly set up
	return np.argmax(policy)

class MonteCarlo():

	def __init__(self, board, player) -> None:
		self.board = board
		self.player = player
	 
	# main function for the Monte Carlo Tree Search
	def search(self) -> int:
		#print('\nStarting search at the following board state \n\n', self.board)
		 # fetch policy and value from neural network given board state
		policy, value = model.forward(self.board)
		replay_buffer = self.board, policy, value
		
		policy = live_game.trim_and_normalize(policy)
		
		policy_target = []
		search_result = {}
		for i in range(len(policy)): # Iterate through all possible moves
			if policy[i]:
				column = i
				empirical_mean_value = simulate(board=np.array(self.board), action=column, player=self.player)
				search_result[i] = empirical_mean_value
				policy_target.append(empirical_mean_value)
			else:
				policy_target.append(0)
		print(f'Search results: {search_result}')
		best_move = max(search_result, key=search_result.get)
		
		value_target = np.array(np.sum(policy_target)/live_game.shape[1])
		policy_target = np.array(policy_target)/np.linalg.norm(policy_target)
		replay_buffer = replay_buffer, policy_target, value_target
		print(replay_buffer)
		#print(policy_target)

		return best_move

	def play_move(self, action):
		#print(f'Making move {action} as player {self.player}')
		live_game.move(column=action, player=self.player)
		self.player *= -1
		#print('Resulting board state:', self.board)

	def _backpropagate(self, sequence, dict):
		for i in range(len(sequence)-1, -1, -2):
			id = sequence[i]
			dict[id][0] += 1

# ---- test run section below ---- #
player = 1
live_game = env.ConnectFour()
mcts = MonteCarlo(live_game.get_board(), player=player)

while not live_game.is_game_over():
	beset_move = mcts.search()
	mcts.play_move(beset_move)
reward = live_game.is_win_state()[1]
print(f'\nThe winner is {reward} with board state:\n{live_game.board}')

'''
history = []

number_of_games = 100
n = number_of_games
game_counter = 0
while number_of_games > 0:
	player = 1
	live_game = env.ConnectFour()
	mcts = MonteCarlo(live_game.get_board(), player=player)
	number_of_moves = 0

	while not live_game.is_game_over():
		#print('Running live game')
		best_move = mcts.search()
		mcts.play_move(best_move)
		number_of_moves += 1
	reward = live_game.is_win_state()[1]
	history.append(number_of_moves)
	#print(f'The winner is {reward} after {number_of_moves} moves')
	
	player *= -1
	number_of_games -= 1
	game_counter += 1
	completion = game_counter/n
	print('Progress:', round(completion*100),'%')
	if game_counter > n-3:
		print(live_game.board)

x = np.linspace(0, len(history), len(history))
y = history

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.savefig('simple plot for learning rate')
'''

outfile = open(filename, 'wb')
pickle.dump(node_dictionary, outfile)
outfile.close