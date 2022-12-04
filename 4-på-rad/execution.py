import monte_carlo as mc
import training_environment as tre
import env

def play_one_game(game, method, show_result=False):
	while not game.is_game_over():
		method.search_and_play()
	reward = game.is_win_state()[1]
	if show_result:
		print(f'\nThe winner is {reward} with board state:\n{game.get_board()}')
		print(f'training_data: {method.training_data}')

def single_iteration():
    network_name = './new_connect_4_brain'
    game = env.ConnectFour()
    neural_network = tre.load_network(network_name, debug=True)
    node_dictionary = mc.load_nodes(filename='nodes', debug=True)
    training_data = mc.load_training_data(filename='training_data', debug=True)
    method = mc.MonteCarlo(game=game, network=neural_network, node_dictionary=node_dictionary, training_data=training_data)

    # Playing one game and extracting update node_dictionary and training_data
    play_one_game(game, method, show_result=True)

    training_data = mc.reshape_training_data(method.training_data, debug=True)
    node_dictionary = method.node_dictionary
    
    mc.save_node_dictionary(dict=node_dictionary, filename='nodes', debug=True)
    mc.save_training_data(data=training_data, filename='training_data_backup', append=False, debug=True)
    
    tre.train(neural_network, training_data, epochs=3, debug=True)
    tre.save_network(neural_network, name=network_name, debug=True)

    #mc.save_training_data(data=training_data, filename='training_data')
    #mc.reset_training_data(filename='training_data')

def multiple_iterations(number_of_iterations, counter=False):
    while number_of_iterations > 0:
        single_iteration()
        number_of_iterations -= 1
        if counter:
            print(f'{number_of_iterations} more iterations')

for i in range(100):
    print(f'\nStage {i+1}')
    multiple_iterations(10)

#training_data = mc.load_training_data(filename='training_data_backup', debug=False)
#training_data = mc.reshape_training_data(training_data, debug=True)
#neural_network = tre.load_network(name='./connect_4_brain', debug=True)
#tre.train(network=neural_network, training_data=training_data, epochs=5)