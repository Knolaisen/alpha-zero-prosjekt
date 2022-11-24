import numpy as np
from neural_network import model
import env
from node import node_dictionary

game = env.ConnectFour()


def trim_and_normalize(array):
	# Replacec illegal moves with value 0
	illegal_moves = game.get_illegal_moves()
	if illegal_moves:
		for i in illegal_moves:
			array[i] = 0
		# Normalize policy (NB!: does not sum to 1 it seems)
		return array/np.linalg.norm(array)
	return array


class MonteCarlo():

	def __init__(self) -> None:
		self.board = game.get_board()	

	def _select(self, policy): #TODO include UBC & value and make sure it's correctly set up
		return np.argmax(policy)

	def _backpropagate(self, sequence, dict):
		for i in range(len(sequence)-1, -1, -2):
			id = sequence[i]
			dict[id][0] += 1
		
	def _simulate(self, action):
		player = 1
		print('action:', action)
		simulation_game = env.ConnectFour()
		simulation_game.board = self.board
		simulation_game.move(action, player)
		
		id_buffer = [] # keep track on board states visited

		while not simulation_game.is_win_state()[0]:
			
			# Enter if board state has been visited before. Dictionary values: [wins, visits]
			id = hash(simulation_game.board.data.tobytes())
			id_buffer.append(id)
			if id in node_dictionary:
				node_dictionary[id][1] += 1 # Adding 1 on number of visits at this state
			else:
				node_dictionary[id] = [0,1] # Making a dictionary entry starting with 1 number of visits
			
			policy, value = model.forward(simulation_game.board)
			policy = trim_and_normalize(policy)
			action = self._select(policy)
			player *= -1
			simulation_game.move(action, player)

		winning_player = simulation_game.is_win_state()[1]
		self._backpropagate(id_buffer, node_dictionary)

		print(f'\n Winner is {winning_player}\n\n{simulation_game.board}')
		#print(f'Node_dictionary: {node_dictionary}')

		return winning_player

	
	# main function for the Monte Carlo Tree Search
	def search(self):
		print('\nStarting search \n\n', self.board)
		 # fetch policy and value from neural network given board state
		policy, value = model.forward(self.board)
		replay_buffer = self.board, policy, value
		
		policy = trim_and_normalize(policy)
		
		search_results = []
		print(policy)
		for i in range(len(policy)): # Iterate through all possible moves
			if policy[i]:
				column = i
				action = column
				print('action', action)
				winning_player = self._simulate(action)
				search_results.append(winning_player)
		
		replay_buffer = replay_buffer, search_results

		return search_results

mcts = MonteCarlo()
mcts.search()