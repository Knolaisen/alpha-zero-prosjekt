import env
import numpy as np
from neural_network import model
from simulate import simulate

node_dictionary = {}

def trim_and_normalize(array):
	# Replacec illegal moves with value 0
	illegal_moves = game.get_illegal_moves()
	if illegal_moves:
		for i in illegal_moves:
			array[i] = 0
		return array/np.linalg.norm(array) # Return normalized policy (NB!: does not sum to 1 it seems)
	return array

class MonteCarlo():

	def __init__(self, player) -> None:
		self.board = game.get_board()
		self.player = player

	# main function for the Monte Carlo Tree Search
	def search(self) -> int:
		print('\nStarting search at the following board state \n\n', self.board)
		 # fetch policy and value from neural network given board state
		policy, value = model.forward(self.board)
		replay_buffer = self.board, policy, value
		
		policy = trim_and_normalize(policy)
		
		policy_target = []
		search_result = {}
		print(f'Trimmed policy: {policy}')
		for i in range(len(policy)): # Iterate through all possible moves
			if policy[i]:
				column = i
				empirical_mean_value = simulate(board=self.board, action=column, player=self.player)
				search_result[i] = empirical_mean_value
				policy_target.append(empirical_mean_value)
			else:
				policy_target.append(0)
		print(f'Search results: {search_result}')
		best_move = max(search_result, key=search_result.get)
		
		policy_target = np.array(policy_target)
		replay_buffer = replay_buffer, policy_target

		return best_move

	def play_move(self, action):
		print(f'Making move as player {self.player}')
		game.move(column=action, player=self.player)
		self.player *= -1
		print('Resulting board state:', self.board)

	def _backpropagate(self, sequence, dict):
		for i in range(len(sequence)-1, -1, -2):
			id = sequence[i]
			dict[id][0] += 1


game = env.ConnectFour()
player = 1
mcts = MonteCarlo(player=player)

while not game.is_win_state()[0]:
	best_move = mcts.search()
	mcts.play_move(best_move)
reward = game.is_win_state()[1]
print(f'\n\nThe winner is {reward} with board \n{board}')