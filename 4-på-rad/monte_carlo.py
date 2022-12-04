import env
import numpy as np
import pickle

# Dictionary storing all visited board states and their repective empirical value and visit count.
def load_nodes(filename='nodes', debug=False) -> dict:
	try:
		with open(filename, 'rb') as infile:
			node_dictionary = pickle.load(infile)
			if debug:
				print('Loading node dictionary at the length of', len(node_dictionary))
			return node_dictionary
	except (OSError, IOError) as e:
		if debug:
			print('No node dictionary found. Returning empty dictionary')
		return {}
		
# Array for storing all training data: board state, policy target and value target respectively
def load_training_data(filename='training_data', debug=False) -> object:
	try:
		training_data = np.array([], dtype='object')
		infile = open(filename, 'rb')
		infile.seek(0)
		while True:
			try:
				training_data = np.append(training_data, pickle.load(infile))
				if debug:
					print('Loading training data at the shape of', np.shape(training_data))
			except EOFError as e:
				print(f'No more input to read. Closing file {filename} and returning training data in the shape of {np.shape(training_data)}')
				infile.close()
				return training_data
	except (OSError, IOError) as e:
		if debug:
			print('No training data found. Returning empty array')
		return np.array([], dtype='object')

def simulate(board, network, action, node_dictionary) -> float:
	
	simulation_game = env.ConnectFour()
	simulation_game.board = board
	#print(f'\nSimulating game starting with the following board')
	#print(simulation_game.board)
	#print(f'Simulating move at column {action}')
	simulation_game.move(action)
	
	# Store resulting board state in node_dictionary to later update its win-rate
	start_id = hash(simulation_game.get_board().data.tobytes())
	id_buffer = [] # keep track on board states visited
	id_buffer.append(start_id)
	if start_id in node_dictionary:
		node_dictionary[start_id][1] += 1 # Adding 1 on number of visits at this state
	else:	
		node_dictionary[start_id] = [0,1] # Making a dictionary entry starting with 1 number of visits
	
	while not simulation_game.is_game_over():
		#print(f'Game counter: {simulation_game.counter}')

		# simulating opponents move
		if simulation_game.counter % 2 == 1:
			simulation_game.switch_sides()
			policy = network.forward(simulation_game.get_board(), only_policy=True)
			policy = simulation_game.trim_and_normalize(policy)
			action = np.argmax(policy)
			simulation_game.move(action)
			simulation_game.switch_sides()
			#print(f'Board state after opponent move:\n {simulation_game.board}')
			continue

		# simulating own move
		else:
			id = hash(simulation_game.get_board().data.tobytes())

			# Enter if board state has been visited before. Dictionary values: [wins, visits]
			if simulation_game.counter > 2:
				id_buffer.append(id)
				if id in node_dictionary:
					node_dictionary[id][1] += 1 # Adding 1 on number of visits at this state
				else:
					node_dictionary[id] = [0,1] # Making a dictionary entry starting with 1 number of visits

			policy = network.forward(simulation_game.get_board(), only_policy=True)
			policy = simulation_game.trim_and_normalize(policy)
			action = select(policy)
			simulation_game.move(action)
			#print(f'Simulating move at column {action}\nResulting board state:\n{simulation_game.board}')
			
	reward = simulation_game.is_win_state()[1]
	#print(f'\nThe simulated winner is {reward} with board \n{simulation_game.board}')

	if reward:
		for id in id_buffer:
			node_dictionary[id][0] += reward

	empirical_value, number_of_visits = node_dictionary[id_buffer[0]]
	
	return empirical_value, number_of_visits, node_dictionary

def get_ucb_score(win_ratio_child, n_parent, n_child) -> float:
	constant = np.sqrt(2)
	return win_ratio_child + constant * np.sqrt( np.log(n_parent) / n_child )

class MonteCarlo():

	def __init__(self, game, network, node_dictionary, training_data, id_buffer=[], use_only_NN=False, interactive=[0, 0]) -> None:
		self.use_only_NN = use_only_NN
		self.interactive = interactive
		self.node_dictionary = node_dictionary
		self.training_data = training_data
		self.id_buffer = id_buffer
		self.game = game
		self.network = network
		
	# main function for the Monte Carlo Tree Search
	def search_and_play(self) -> None:
		
		self.game.counter += 1

		player_turn = self.interactive[1]
		if player_turn:
			print(f'\nYou are now playing against the AI!')
			self.game.switch_sides()
			print(f'\nCurrent board state:\n {self.game.get_board()}')
			available_moves = self.game.get_available_moves()
			best_move = int(input(f'\nChoose your column! Available moves: {available_moves}\n>'))
			while best_move not in available_moves:
				print(f'You attempted to play on column {best_move}, however this move is not valid')
				best_move = int(input(f'\nChoose your column! Available moves: {available_moves}\n> '))
			self.interactive[1] = False
			self.game.move(best_move)
			self.game.switch_sides()
			return None
		
		print(f'{self.game.counter}: board state: \n{self.game.get_board()}')

		raw_policy = self.network.forward(self.game.get_board(), only_policy= True)
		print(f'Raw-policy: {raw_policy}')
		policy = self.game.trim_and_normalize(raw_policy)
		print(f'trim and normalized policy: {policy}')

		if self.use_only_NN:
			best_move = np.argmax(policy)
			self.game.move(best_move)
			return None

		parent_id = hash(self.game.get_board().data.tobytes())
		self.id_buffer.append(parent_id)
		if parent_id in self.node_dictionary:
			n_visits_parent = self.node_dictionary[parent_id][1] = self.node_dictionary[parent_id][1] + 1 # Fetch numer of visits and add 1 on number of visits at this state
		else:	
			self.node_dictionary[parent_id] = [0,1] # Making a dictionary entry starting with 1 number of visits
			n_visits_parent = 1

		if len(self.game.get_available_moves()) > 1: # enter if there are more than 1 available move
			print('Entering for-loop for simulating games ...')
			updated_policy_target = np.array([])
			win_rates = []
			search_result = {}
			for i in range(len(policy)): # Iterate through all possible moves
				if i in self.game.get_illegal_moves():
					updated_policy_target = np.append(updated_policy_target, values=0)
				else:
					column = i
					empirical_value, n_visits_child, self.node_dictionary = simulate(board=self.game.get_board(), network=self.network, action=column, node_dictionary=self.node_dictionary)
					empirical_mean_value = empirical_value/n_visits_child
					win_rates.append(empirical_mean_value)
					ucb_score = get_ucb_score(win_ratio_child= empirical_mean_value, n_parent= n_visits_parent, n_child= n_visits_child)
					search_result[i] = ucb_score
					print(f'UCB score for column {column} is {ucb_score}')
					updated_policy_target = np.append(updated_policy_target, values=ucb_score)
			print(f'Search results: {search_result}')
			if not (updated_policy_target==np.zeros(7)).all():
				policy_target = updated_policy_target
				best_move = max(search_result, key=search_result.get)
				policy_norm = np.linalg.norm(policy_target)
				policy_target = policy_target/policy_norm
				value_target = np.array([np.sum(win_rates)/7], dtype='float32')
			else: # Enter if search result yields 0 for all moves
				policy_target = policy
				value_target = np.array([0])
				best_move = np.argmax(policy_target)
		else: # enter if there is only one move available
			policy_target = policy
			best_move = np.argmax(policy_target)
			empirical_value, n_visits_child, self.node_dictionary = simulate(board=self.game.get_board(), network=self.network, action=best_move, node_dictionary=self.node_dictionary)
			value_target = np.array([empirical_value/n_visits_child])
		
		#print(f'Policy target: {policy_target}, value target: {value_target}')		
		if value_target > 1:
			raise Exception('WARNING: value_target lies outside of legal range [-1, 1]')

		#print('raw_value: ', raw_value)
		#print('target_value:', value_target)

		replay_buffer = self.game.get_board(), policy_target, value_target
		replay_buffer = np.array([replay_buffer], dtype='object')
		#print(f'replay buffer:\n{replay_buffer}')
		#self.training_data = np.append(self.training_data, replay_buffer)
		self.training_data = np.append(arr=self.training_data, values=replay_buffer)
		#print(f'training_data: {self.training_data}')

		save_training_data(replay_buffer, filename='training_data', debug=True, append=True)

		if self.interactive[0] and not self.interactive[1]:
			self.interactive[1] = True

		if best_move == -1:
			raise Exception('WARNING: best_move parameter initialized to -1 was not updated from the simulationg games.')

		self.game.move(best_move)
		self.game.switch_sides()
		return None

def save_node_dictionary(dict, filename='nodes', debug=False):
	with open(filename, 'wb') as outfile:
		pickle.dump(dict, outfile)
		if debug:
			print('Saving node dictionary at the length of', len(dict))

def reset_node_dictionary(): #TODO FIX: reset funksjonen fungerer ikke enda, så filene må slettes manuelt
	empty_node_dictionary = {}
	file = open(filename_nodes, 'wb')
	pickle.dump(empty_node_dictionary, file)
	file.close()

def reshape_training_data(data, debug=False):
	columns = 3
	rows = int(len(data)/columns)
	training_data = np.reshape(data, newshape=(rows, columns))
	if debug:
		print(f'Reshaped training data to shape: {np.shape(training_data)}')
	return training_data

def save_training_data(data, filename='training_data', debug=False, append=False):
	if append:
		if debug:
			file = open(filename, 'ab+')
			pickle.dump(data, file)
			print(f'Appended training_data of shape {np.shape(data)}')
	
			file.seek(0)
			training_data = np.array([], dtype='object')
			while True:
				try:
					training_data = np.append(training_data, pickle.load(file))
				except EOFError as e:
					print(f'Closing file with updated shape of {np.shape(training_data)}')
					file.close()
					break
		else:
			with open(filename, 'ab') as outfile:
				pickle.dump(data, outfile)
	else:
		with open(filename, 'wb') as outfile:
			pickle.dump(data, outfile)
		if debug:
			print('Dumped training data at the shape of', np.shape(data))
	

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

def reset_training_data(filename='training_data'): # TODO FIX: reset funksjonen fungerer ikke enda, så filene må slettes manuelt
	print(f'Resetting {filename}')
	empty_training_data = np.array([], dtype='object')
	file = open(filename, 'wb')
	pickle.dump(empty_training_data, file)
	file.close()

def play_one_game(game, method, show_result=False):
	
	while not game.is_game_over():
		method.search_and_play(game=game)
	if show_result:
		reward = game.is_win_state()[1]
		print(f'\nThe winner is {reward} with board state:\n{game.get_board()}')
		print(f'training_data: {method.training_data}')

def play_multiple_games(number_of_games, save=False):
	number_of_games = number_of_games
	n = number_of_games
	game_counter = 0
	while number_of_games > 0:
		play_one_game(save=save)
		number_of_games -= 1
		game_counter += 1
		completion = game_counter/n
		print('Progress:', round(completion*100),'%')
		if game_counter > n-3:
			print(game.get_board())
