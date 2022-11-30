import env
import numpy as np
from neural_network import model
import pickle

# Dictionary storing all visited board states and their repective empirical value and visit count.
filename_nodes = 'nodes'
try:
	infile_nodes = open(filename_nodes, 'rb')
	node_dictionary = pickle.load(infile_nodes)
	infile_nodes.close()
except (OSError, IOError) as e:
	node_dictionary = {}

# Array for storing all training data: board state, policy target and value target respectively 
filename_training = 'training_data'
try:
	infile_training = open(filename_training, 'rb')
	training_data = pickle.load(infile_training)
	infile_training.close()
except (OSError, IOError) as e:
	training_data = np.array([], dtype='object')

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
		raw_policy, raw_value = model.forward(self.board)
		policy = live_game.trim_and_normalize(raw_policy)
		
		policy_target = np.array([])
		search_result = {}
		for i in range(len(policy)): # Iterate through all possible moves
			if policy[i]:
				column = i
				empirical_mean_value = simulate(board=np.array(self.board), action=column, player=self.player)
				search_result[i] = empirical_mean_value
				policy_target = np.append(policy_target, values=empirical_mean_value)
			else:
				policy_target = np.append(policy_target, values=0)
		#print(f'Search results: {search_result}')
		best_move = max(search_result, key=search_result.get)
		
		value_target = np.array([np.sum(policy_target)/live_game.shape[1]], dtype='float32')
		policy_target = policy_target/np.linalg.norm(policy_target)

		#print('raw_value: ', raw_value)
		#print('target_value:', value_target)

		replay_buffer = self.board, policy_target, value_target
		replay_buffer = np.array([replay_buffer], dtype='object')
		
		global training_data
		training_data = np.append(training_data, replay_buffer)

		return best_move

	def play_move(self, action) -> None:
		#print(f'Making move {action} as player {self.player}')
		live_game.move(column=action, player=self.player)
		self.player *= -1
		#print('Resulting board state:', self.board)

	def _backpropagate(self, sequence, dict) -> None:
		for i in range(len(sequence)-1, -1, -2):
			id = sequence[i]
			dict[id][0] += 1

def show_training_data():
	columns = 3
	global training_data
	rows = int(len(training_data)/columns)
	training_data = np.reshape(training_data, newshape=(rows, columns))

	print('\nTraining data:\n')
	for i in range(3):
		print('Row',i, ':', training_data[i])
	for i in range(len(training_data)-1,len(training_data)-3, -1):
		print('Row',i, ':', training_data[i])

def save_node_dictionary():
	print('Saving node dictionary at the length of', len(node_dictionary))
	outfile_nodes = open(filename_nodes, 'wb')
	pickle.dump(node_dictionary, outfile_nodes)
	outfile_nodes.close()

def save_training_data():
	print('Saving training data at the shape of', np.shape(training_data))
	outfile_games = open(filename_training, 'wb')
	pickle.dump(training_data, outfile_games)
	outfile_games.close()

def play_one_game(player):
	player = player
	global live_game
	live_game = env.ConnectFour()
	mcts = MonteCarlo(live_game.get_board(), player=player)

	while not live_game.is_game_over():
		beset_move = mcts.search()
		mcts.play_move(beset_move)
	reward = live_game.is_win_state()[1]
	print(f'\nThe winner is {reward} with board state:\n{live_game.board}')

def play_multiple_games(number_of_games):
	player = 1
	number_of_games = number_of_games
	n = number_of_games
	game_counter = 0
	while number_of_games > 0:
		play_one_game(player=player)
		player *= -1
		number_of_games -= 1
		game_counter += 1
		completion = game_counter/n
		print('Progress:', round(completion*100),'%')
		if game_counter > n-3:
			print(live_game.board)

# ---- test run section below ---- #

play_multiple_games(number_of_games= 2)

show_training_data()

save_node_dictionary()
save_training_data()