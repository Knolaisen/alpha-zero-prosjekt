import env
import numpy as np
from neural_network import model

def trim_and_normalize(array, game):
	# Replacec illegal moves with value 0
	illegal_moves = game.get_illegal_moves()
	if illegal_moves:
		for i in illegal_moves:
			array[i] = 0
		return array/np.linalg.norm(array) # Return normalized policy (NB!: does not sum to 1 it seems)
	return array

def select(policy): #TODO include UBC & value and make sure it's correctly set up
	return np.argmax(policy)

# Simulates game from given board state and returns updated empirical mean value for board state after finished simulation
def simulate(board, player) -> float: 

    simulation_game = env.ConnectFour()
    simulation_game.board = board
    #print(f'\nSimulating game starting with player {player} with clean board')
    
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
        policy = trim_and_normalize(policy, simulation_game)
        action = select(policy)
        simulation_game.move(action, player)
        player *= -1
        
    reward = simulation_game.is_win_state()[1]
    #print(f'\nThe winner is {reward} with board \n{board}')
    
    for id in id_buffer:
        node_dictionary[id][0] += reward
        reward *= -1

    empirical_value, number_of_visits = node_dictionary[id_buffer[0]]
    empirical_mean_value = empirical_value/number_of_visits

    return empirical_mean_value

def test():
    global node_dictionary
    node_dictionary = {}
    number_of_games = 100
    player = 1

    while number_of_games > 0:
        board = np.zeros((6,7)) 
        simulate(board, player)
        number_of_games -= 1
        player *= -1

    empirical_value, number_of_visits = node_dictionary[hash(np.zeros((6,7)).data.tobytes())]
    print(f'\nEmpirical value: {empirical_value}\nNumber of visits: {number_of_visits}')
    empirical_mean_value = empirical_value/number_of_visits
    print(f'Empirical mean value for clean board state is {empirical_mean_value}')

    for key in node_dictionary:
        if node_dictionary[key][1] > 1:
            print(node_dictionary[key])
#test()