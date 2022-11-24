import neural_network as NN
import env

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")

class MonteCarlo():

	def __init__(self) -> None:
		self.board = env.ConnectFour.get_board()	

	# main function for the Monte Carlo Tree Search
	def monte_carlo_tree_search(root):

#		policy, value = NN.forward(self.board)
#		for possible_action in policy:
			

		while resources_left(time, computational power):
			leaf = traverse(root)
			simulation_result = rollout(leaf)
			backpropagate(leaf, simulation_result)
			
		return best_child(root)

	# function for node traversal
	def traverse(node):
		while fully_expanded(node):
			node = best_uct(node)
			
		# in case no children are present / node is terminal
		return pick_unvisited(node.children) or node

	# function for the result of the simulation
	def rollout(node):
		while non_terminal(node):
			node = rollout_policy(node)
		return result(node)

	# function for randomly selecting a child node
	def rollout_policy(node):
		return pick_random(node.children)

	# function for backpropagation
	def backpropagate(node, result):
		if is_root(node) return
		node.stats = update_stats(node, result)
		backpropagate(node.parent)

	# function for selecting the best child
	# node with highest number of visits
	def best_child(node):
		pick child with highest number of visits
