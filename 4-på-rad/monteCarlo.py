import environment
import numpy as np
import dataHandler
import hashlib
from copy import deepcopy

class MonteCarlo():

	def __init__(self, network, node_dictionary, debug=False) -> None:
		self.network = network
		self.node_dictionary = node_dictionary
		self.training_data = np.array([], dtype=object)
		self.id_buffer = [] # Buffer tracking board states in sequence from start to finish.

		if debug:
			print(f'Initiating method class MonteCarlo with node dictionary of length {len(self.node_dictionary)} and the following policy&value-network:\n{self.network}')
	
	def _trim_and_softmax(self, array, illegal_moves):
		for i in illegal_moves: # Replacec illegal moves with -infinity to get value 0 from softmax.
			array[i] = float('-inf')
		return np.exp(array)/np.sum(np.exp(array)) # softmax
		
	# main function for the Monte Carlo Tree Search
	def search(self, board, available_moves, illegal_moves, save_training_data: tuple = (True, 'training_data'), debug=False) -> int:
		#print(f'\n\nInitiating search at board state:\n{board}\n, which is of the following type: {type(board)}')
		
		board, available_moves, illegal_moves = deepcopy(board), deepcopy(available_moves), deepcopy(illegal_moves) # Just for safety

		raw_policy = self.network.forward(board, only_policy= True)
				
		policy = self._trim_and_softmax(
			array= raw_policy,
			illegal_moves= illegal_moves,
			)
		
		parent_id = hashlib.md5(board.data.tobytes()).hexdigest()
		#print(f'parent id:{parent_id}')
		self.id_buffer.append(parent_id)
		if parent_id in self.node_dictionary.keys():
			#print('Board state already visited')
			self.node_dictionary[parent_id][1] += 1
			n_visits_parent = self.node_dictionary[parent_id][1]  # Fetch numer of visits and add 1 on number of visits at this state
		else:	
			#print('Board state never visited before')
			self.node_dictionary[parent_id] = [0,1] # Making a dictionary entry starting with 1 number of visits
			n_visits_parent = 1

		if len(available_moves) == 1: # enter if ther is only 1 more available move
			policy_target = policy
			best_move = np.argmax(policy_target)
			simulation_board = deepcopy(board)
			empirical_value_child, n_visits_child = self._simulate(simulation_board, best_move, parent_id)
			value_target = np.array([empirical_value_child/n_visits_child], dtype='float32')
			#print(f'Only one available move. Resulting policy target: {policy_target}, and value target: {value_target}')
			
		else: # enter if there are more than 1 available moves
			#print('Entering for-loop for simulating games ...')
			updated_policy_target = np.full(shape=7, fill_value='-inf', dtype='float32') # Fill with -infinity to yield value 0 from softmax
			win_rate = np.zeros(2) # [accumulating win_rate, number of additions (later used to find the mean)]
			for action in available_moves: # Iterate through all possible moves				
				# Simulate an entire game guided by the policy network and update policy target with ucb_score
				simulation_board = deepcopy(board)
				empirical_value_child, n_visits_child = self._simulate(simulation_board, action, parent_id)

				win_ratio_child = float(empirical_value_child/n_visits_child)
				win_rate[0] += win_ratio_child
				win_rate[1] += 1

				ucb_constant = np.sqrt(2)
				ucb_score = win_ratio_child + ucb_constant * np.sqrt( np.log(n_visits_parent) / n_visits_child )
				updated_policy_target[action] = ucb_score
			#print(f'Search results (policy target before softmax): {updated_policy_target}')
			policy_target = np.exp(updated_policy_target)/np.sum(np.exp(updated_policy_target))
			value_target = np.array([win_rate[0]/win_rate[1]], dtype='float32')
			best_move = np.random.choice(np.where(policy_target == policy_target.max())[0])
		
		#print(f'Policy target: {policy_target}, value target: {value_target}')

		replay_buffer = deepcopy(board), deepcopy(policy_target), deepcopy(value_target)
		replay_buffer = np.array([replay_buffer], dtype='object')
		#print(f'replay buffer:\n{replay_buffer}')

		# Exceptions
		if abs(value_target) > 1:
			raise Exception(f'WARNING: value_target lies outside of bound range [-1, 1]\nReplay buffer:\n{replay_buffer}')
		if np.count_nonzero(abs(policy_target)>0) > len(available_moves):
			exception_message = f'WARNING: flawed policy target. Replay buffer: \n{replay_buffer}'
			raise Exception(exception_message)

		
		if save_training_data[0]:
			dataHandler.save_training_data(
				data= replay_buffer,
				filename= save_training_data[1],
				debug= False,
				append= True,
				)

		self.training_data = np.append(self.training_data, replay_buffer)

		return best_move

	def _simulate(self, board, action, parent_id) -> tuple:
		
		self.node_dictionary[parent_id][1] += 1
		simulation_game = environment.ConnectFour(board=board)
		
		#print(f'\n\nSimulating game starting with the following board:\n{board}\n, which has the respective board info: {self.node_dictionary[hash(board.data.tobytes())]}')
		#print(f'Simulating move at column {action} ...')
		simulation_game.move(action)

		# Store resulting board state in node_dictionary to later update its win-rate
		start_id = hashlib.md5(simulation_game.board.data.tobytes()).hexdigest()
		simulation_id_buffer = [parent_id] # keep track on board states visited
		simulation_id_buffer.append(start_id)
		if start_id in self.node_dictionary.keys():
			self.node_dictionary[start_id][1] += 1 # Adding 1 on number of visits at this state
		else:	
			self.node_dictionary[start_id] = [0,1] # Making a dictionary entry starting with 1 number of visits
		#print(f'Initiating simulation at board state: \n{simulation_game.board}, which has the following board info: {self.node_dictionary[start_id]}')

		while not simulation_game.is_game_over():
			#print(f'Simulation game counter: {simulation_game.counter}')

			# simulating opponents move
			if simulation_game.counter % 2 == 1:
				simulation_game.switch_sides()
				policy = simulation_game.trim(
					self.network.forward(simulation_game.board, only_policy=True),
				)
				action = np.random.choice(np.where(policy == policy.max())[0])
				simulation_game.move(action)
				simulation_game.switch_sides()
				#print(f'Board state after opponent move:\n {simulation_game.board}')
				continue

			# simulating own move
			else:
				id = hashlib.md5(simulation_game.board.data.tobytes()).hexdigest()
				# Enter if board state has been visited before. Dictionary values: [wins, visits]
				if simulation_game.counter > 2:
					simulation_id_buffer.append(id)
					if id in self.node_dictionary.keys():
						self.node_dictionary[id][1] += 1 # Adding 1 on number of visits at this state
					else:
						self.node_dictionary[id] = [0,1] # Making a dictionary entry starting with 1 number of visits

				policy = self.network.forward(simulation_game.board, only_policy=True)
				#policy = simulation_game.trim_and_softmax(policy)
				policy = simulation_game.trim(policy)
				action = np.random.choice(np.where(policy == policy.max())[0])
				simulation_game.move(action)
				#print(f'Simulating move at column {action}\nResulting board state:\n{simulation_game.board}')
				
		reward = simulation_game.is_win_state()[1]
		#print(f'\nThe simulated winner is {reward}')

		if reward: # Only backpropagate node values if the reward is something else than zero.
			for id in simulation_id_buffer:
				self.node_dictionary[id][0] += reward
	
		empirical_value = self.node_dictionary[start_id][0]
		number_of_visits = self.node_dictionary[start_id][1]
		#print(f'Closing simulation with board info: {self.node_dictionary[start_id]}')
		#print(f'empirical value: {empirical_value}, number of visits: {number_of_visits}')
		
		return empirical_value, number_of_visits
		
	def backpropagate(self, reward: int) -> None:
		for id in self.id_buffer:
			self.node_dictionary[id][0] += reward
		self.id_buffer = []
