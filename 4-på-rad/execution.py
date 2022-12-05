import numpy as np
import monteCarlo
import trainingEnvironment
import environment
import dataHandler
import hashlib

def play_supervised_move(game): 
    print(f'\nYou are now playing against the AI!')
    game.switch_sides()
    print(f'\nCurrent board state:\n {game.get_board()}')
    available_moves = game.get_available_moves()
    supervised_move = int(input(f'\nChoose your column! Available moves: {available_moves}\n>'))
    while supervised_move not in available_moves:
        print(f'You attempted to play on column {supervised_move}, however this move is not valid')
        supervised_move = int(input(f'\nChoose your column! Available moves: {available_moves}\n> '))
    game.move(supervised_move)
    game.switch_sides()
    return None

def play(number_of_games = 1):
    network_name = './connect_4_brain_new.pth'
    neural_network = trainingEnvironment.load_network(network_name, debug=False)
    neural_network.eval()
    node_dictionary = dataHandler.load_nodes(filename='visited_nodes_new', debug=True)
    
    method = monteCarlo.MonteCarlo(
        network= neural_network,
        node_dictionary= node_dictionary,
        debug= False
    )
    game = environment.ConnectFour(debug=False)   
    number_of_games_left = number_of_games 
    while number_of_games_left > 0:
        game.reset()
        #print(f'{number_of_games_left} number of games left ...')
        
        if number_of_games_left % 2 == 1: # Alternate between opening the game, and answering to opponents opening.
            policy = neural_network.forward(game.board, only_policy= True) 
            opponent_opening_move = np.random.choice(np.where(policy == policy.max())[0])
            game.move(opponent_opening_move)
            game.counter = 0 # Reset counter
            game.switch_sides()

        #print(f'\nStarting game at counter {game.counter} and board state: \n{game.board}')
        while not game.is_game_over():
            #print(f'After {game.counter} moves, the following board state is: \n{game.board}')

            if game.counter % 2 == 1:
                game.switch_sides()
                best_move = method.search(
                    board= game.get_board(),
                    available_moves= game.get_available_moves(),
                    illegal_moves= game.get_illegal_moves(),
                    save_training_data= (True, 'training_data_for_player_-1_backup_new')
                )
                game.move(best_move)
                #print(f'Player -1 choses column {best_move}.')
                win_state = game.is_win_state()
                if win_state[0]:
                    reward = win_state[1]
                game.switch_sides()
                continue
            if game.counter % 2 == 0:
                best_move = method.search(
                    board= game.get_board(),
                    available_moves= game.get_available_moves(),
                    illegal_moves= game.get_illegal_moves(),
                    save_training_data= (True, 'training_data_for_player_1_backup_new')
                )
                game.move(best_move)
                #print(f'Player 1 choses column {best_move}')
                win_state = game.is_win_state()
                if win_state[0]:
                    reward = win_state[1]
                continue
        reward = game.is_win_state()[1]
        method.backpropagate(reward)
        #print(f'The winner is {reward} at board state:\n{game.board}')
        number_of_games_left -= 1
        
    print(f'Win rate [wins, number of games] at clean board: {method.node_dictionary[hashlib.md5(np.zeros((6,7)).data.tobytes()).hexdigest()]}')

    dataHandler.save_node_dictionary(
        dict= method.node_dictionary,
        filename= 'visited_nodes_new',
        debug= True,
    )
    training_data = dataHandler.reshape_training_data(
        data= method.training_data,
        debug= False,
    )
    neural_network.train()
    trainingEnvironment.train(
        network= neural_network,
        training_data= training_data,
        epochs= 2,
        debug= True,
    )
    trainingEnvironment.save_network(
        network= neural_network,
        name= './connect_4_brain_new.pth',
        debug= True,
    )
    return None

def iterate(iterations, games_each=1):
    while iterations > 0:
        play(number_of_games= games_each)
        iterations -= 1
        #if iterations % 10 == 0:
        #    print(f'{100-iterations}% done')

def train_exclusively():
    connect_4_brain = trainingEnvironment.load_network(debug=True) 
    trainingEnvironment.train(
        network= connect_4_brain,
        training_data= dataHandler.reshape_training_data(
            data= dataHandler.load_training_data('training_data_for_player_-1_backup', debug=True),
            debug= True
        ),
        epochs=1,
        debug= True
    )
    trainingEnvironment.save_network(
        network= connect_4_brain,
        debug= True
    )
    return None

#train_exclusively()
iterate(iterations= 2**8, games_each= 2**4)
